"""Servidor web local para calibrar meta y lanzar procesamiento."""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bib_timing.bib_gen import BibStyle, bib_to_png_bytes, expand_number_range, generate_bib_zip
from bib_timing.device import probe_device
from bib_timing.finish_line import DetectionZone, FinishLine, Point
from bib_timing.models.registry import (
    create_model,
    get_model,
    list_models,
    load_recognize_known_bibs,
    resolve_weights,
)
from bib_timing.models.train_detect import register_recognize_model, train_detect_model
from bib_timing.ops_review import apply_job_reviews
from bib_timing.pipeline import process_video
from bib_timing.training.dataset import (
    TrainSession,
    export_detect_dataset,
    export_recognize_dataset,
    load_session,
    save_session,
    session_dir,
    training_root,
)
from bib_timing.training.scan import scan_detect_candidates, scan_recognize_candidates
from bib_timing.video_io import VideoOpenError, convert_to_h264, open_video, probe_video_codec

ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = ROOT / "data" / "uploads"
JOB_DIR = ROOT / "output" / "jobs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    cancelled = "cancelled"
    error = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.queued
    message: str = ""
    video_path: str = ""
    output_dir: str = ""
    line: tuple[float, float, float, float] | None = None
    zone: tuple[float, float, float, float] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    preview_url: str | None = None
    csv_url: str | None = None
    live_seq: int = 0
    frame: int = 0
    total_frames: int = 0
    snapshots: list[str] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    detect_model_id: str | None = None
    recognize_model_id: str | None = None
    review_applied: bool = False


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


class ProcessRequest(BaseModel):
    video_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    zx0: float | None = None
    zy0: float | None = None
    zx1: float | None = None
    zy1: float | None = None
    conf: float = Field(0.35, ge=0.05, le=0.95)
    max_frames: int | None = Field(None, ge=1)
    cpu: bool = False
    enable_ocr: bool = True
    detect_model_id: str | None = None
    recognize_model_id: str | None = None


class JobReviewItem(BaseModel):
    track_id: int
    verdict: str  # correct | incorrect | skip
    bib: str | None = None


class JobReviewRequest(BaseModel):
    reviews: list[JobReviewItem]
    refine: bool = True
    cpu: bool = False


class TrainStartRequest(BaseModel):
    video_id: str
    kind: str = "detect"
    conf: float = Field(0.35, ge=0.05, le=0.95)
    max_frames: int | None = Field(None, ge=1)
    frame_stride: int = Field(5, ge=1, le=60)
    max_candidates: int = Field(300, ge=10, le=2000)
    min_score: float = Field(0.14, ge=0.05, le=0.95)
    cpu: bool = False
    # Detector bib para propuestas (refinamiento / reconocimiento)
    detect_model_id: str | None = None


class ModelCreateRequest(BaseModel):
    name: str
    kind: str = "detect"
    parent_id: str | None = None
    notes: str = ""


class ModelTrainRequest(BaseModel):
    epochs: int = Field(40, ge=5, le=200)
    batch: int = Field(8, ge=1, le=32)
    cpu: bool = False
    # Para recognize: vínculo opcional al detector usado
    detect_model_id: str | None = None


class TrainLabelRequest(BaseModel):
    candidate_id: str
    verdict: str = "yes"
    bib: str | None = None


class BibGenerateRequest(BaseModel):
    number: str = "161"
    start: int | None = Field(None, ge=1)
    end: int | None = Field(None, ge=1)
    pad_width: int = Field(0, ge=0, le=8)
    bib_bg: str = "#FFFFFF"
    number_color: str = "#111111"
    marker_color: str = "#111111"
    border_color: str = "#111111"
    race_text: str = "Maratón 2026"
    race_color: str = "#444444"
    page_bg: str = "#F0F0F0"
    width: int = Field(900, ge=400, le=2400)
    height: int = Field(1100, ge=500, le=3000)


def _bib_style_from_req(req: BibGenerateRequest) -> BibStyle:
    return BibStyle(
        bib_bg=req.bib_bg,
        number_color=req.number_color,
        marker_color=req.marker_color,
        border_color=req.border_color,
        race_text=req.race_text,
        race_color=req.race_color,
        page_bg=req.page_bg,
        width=req.width,
        height=req.height,
    )


_train_runtime: dict[str, dict[str, Any]] = {}
_train_lock = threading.Lock()
_model_runtime: dict[str, dict[str, Any]] = {}
_model_lock = threading.Lock()


def create_app() -> FastAPI:
    app = FastAPI(title="AI Checkpoint", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/device")
    def api_device() -> dict[str, Any]:
        info = probe_device()
        return asdict(info)

    @app.get("/api/videos")
    def list_videos() -> dict[str, Any]:
        items = []
        for path in sorted(UPLOAD_DIR.glob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".mp4",
                ".mkv",
                ".avi",
                ".mov",
                ".webm",
            }:
                items.append(
                    {
                        "id": path.name,
                        "name": path.name,
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                    }
                )
        # También clips en data/
        data_dir = ROOT / "data"
        for path in sorted(data_dir.glob("*.mp4")):
            if path.parent == UPLOAD_DIR:
                continue
            items.append(
                {
                    "id": f"data:{path.name}",
                    "name": f"data/{path.name}",
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                }
            )
        return {"videos": items}

    @app.post("/api/upload")
    async def upload_video(
        file: UploadFile = File(...),
        convert_av1: bool = Form(True),
        clip_seconds: float = Form(120.0),
    ) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(400, "Nombre de archivo vacío")
        suffix = Path(file.filename).suffix.lower() or ".mp4"
        video_id = f"{uuid.uuid4().hex[:10]}{suffix}"
        dest = UPLOAD_DIR / video_id
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        codec = probe_video_codec(dest)
        converted = False
        final_path = dest
        warning = None

        if convert_av1 and codec and codec.lower() in {"av1", "av01", "vp9", "vp09"}:
            h264 = UPLOAD_DIR / f"{dest.stem}_h264.mp4"
            try:
                convert_to_h264(dest, h264, seconds=clip_seconds if clip_seconds > 0 else None)
                final_path = h264
                video_id = h264.name
                converted = True
                # borrar original AV1 para no llenar disco
                dest.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                warning = f"No se pudo convertir {codec}: {exc}"

        # validar apertura
        try:
            cap = open_video(final_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except VideoOpenError as exc:
            raise HTTPException(400, str(exc)) from exc

        return {
            "id": video_id,
            "name": file.filename,
            "codec": codec,
            "converted": converted,
            "warning": warning,
            "width": w,
            "height": h,
            "fps": fps,
            "frames": frames,
        }

    @app.get("/api/videos/{video_id}/frame")
    def get_frame(video_id: str, t: float = 0.0) -> FileResponse:
        path = _resolve_video(video_id)
        try:
            cap = open_video(path)
        except VideoOpenError as exc:
            raise HTTPException(400, str(exc)) from exc
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_idx = max(0, int(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(400, "No se pudo leer el frame")

        safe = Path(video_id).name.replace(":", "_")
        out = UPLOAD_DIR / f".frame_{safe}.jpg"
        cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return FileResponse(out, media_type="image/jpeg")

    @app.get("/api/videos/{video_id}/info")
    def video_info(video_id: str) -> dict[str, Any]:
        path = _resolve_video(video_id)
        try:
            cap = open_video(path)
        except VideoOpenError as exc:
            raise HTTPException(400, str(exc)) from exc
        info = {
            "id": video_id,
            "path": str(path),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "codec": probe_video_codec(path),
        }
        cap.release()
        return info

    @app.post("/api/process")
    def start_process(req: ProcessRequest) -> dict[str, Any]:
        path = _resolve_video(req.video_id)
        job_id = uuid.uuid4().hex[:12]
        out_dir = JOB_DIR / job_id
        zone = None
        if None not in (req.zx0, req.zy0, req.zx1, req.zy1):
            zone = (float(req.zx0), float(req.zy0), float(req.zx1), float(req.zy1))
        job = Job(
            id=job_id,
            video_path=str(path),
            output_dir=str(out_dir),
            line=(req.x0, req.y0, req.x1, req.y1),
            zone=zone,
            detect_model_id=req.detect_model_id,
            recognize_model_id=req.recognize_model_id,
            message="En cola",
        )
        with _jobs_lock:
            _jobs[job_id] = job

        thread = threading.Thread(
            target=_run_job,
            args=(job_id, req),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id}

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(404, "Job no encontrado")
            if job.status not in {JobStatus.queued, JobStatus.running}:
                return {"ok": False, "status": job.status.value, "message": "Ya finalizó"}
            job.cancel_requested = True
            job.message = "Deteniendo…"
        return {"ok": True, "status": "stopping"}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(404, "Job no encontrado")
            live_url = None
            live_file = Path(job.output_dir) / "live.jpg"
            if live_file.exists():
                live_url = f"/api/jobs/{job_id}/live?v={job.live_seq}"
            return {
                "id": job.id,
                "status": job.status.value,
                "events": job.events,
                "message": job.message,
                "preview_url": job.preview_url,
                "csv_url": job.csv_url,
                "line": job.line,
                "zone": job.zone,
                "live_url": live_url,
                "live_seq": job.live_seq,
                "frame": job.frame,
                "total_frames": job.total_frames,
                "snapshots": [
                    f"/api/jobs/{job_id}/snapshot/{name}" for name in job.snapshots
                ],
                "pending": job.pending,
                "detect_model_id": job.detect_model_id,
                "recognize_model_id": job.recognize_model_id,
                "review_applied": job.review_applied,
                "review_available": (
                    job.status in {JobStatus.done, JobStatus.cancelled}
                    and len(job.events) > 0
                    and not job.review_applied
                ),
            }

    @app.get("/api/jobs/{job_id}/live")
    def job_live(job_id: str) -> FileResponse:
        path = JOB_DIR / job_id / "live.jpg"
        if not path.exists():
            raise HTTPException(404, "Live frame no disponible")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/jobs/{job_id}/snapshot/{name}")
    def job_snapshot(job_id: str, name: str) -> FileResponse:
        safe = Path(name).name
        path = JOB_DIR / job_id / "snapshots" / safe
        if not path.exists():
            raise HTTPException(404, "Snapshot no encontrado")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/jobs/{job_id}/crop/{name}")
    def job_crop(job_id: str, name: str) -> FileResponse:
        safe = Path(name).name
        path = JOB_DIR / job_id / "crops" / safe
        if not path.exists():
            raise HTTPException(404, "Recorte no encontrado")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/jobs/{job_id}/review")
    def job_review(job_id: str, req: JobReviewRequest) -> dict[str, Any]:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(404, "Job no encontrado")
            if job.status not in {JobStatus.done, JobStatus.cancelled}:
                raise HTTPException(400, "El job aún no finalizó")
            if job.review_applied:
                raise HTTPException(400, "La revisión ya fue aplicada")
            events = list(job.events)
            out_dir = Path(job.output_dir)
            detect_id = job.detect_model_id
            recognize_id = job.recognize_model_id
            video_path = job.video_path

        try:
            summary = apply_job_reviews(
                ROOT,
                out_dir,
                events,
                [r.model_dump() for r in req.reviews],
                video_path=video_path,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

        train_started: list[str] = []
        if req.refine:
            if detect_id and summary.get("detect_labels", 0) > 0:
                with _model_lock:
                    _model_runtime[detect_id] = {
                        "status": "training",
                        "message": "Refinando detector con revisión…",
                    }
                threading.Thread(
                    target=_run_model_train,
                    args=(
                        detect_id,
                        ModelTrainRequest(epochs=30, batch=8, cpu=req.cpu),
                    ),
                    daemon=True,
                ).start()
                train_started.append(detect_id)
            if recognize_id and summary.get("recognize_labels", 0) > 0:
                with _model_lock:
                    _model_runtime[recognize_id] = {
                        "status": "training",
                        "message": "Actualizando reconocimiento…",
                    }
                threading.Thread(
                    target=_run_model_train,
                    args=(
                        recognize_id,
                        ModelTrainRequest(
                            cpu=req.cpu, detect_model_id=detect_id
                        ),
                    ),
                    daemon=True,
                ).start()
                train_started.append(recognize_id)

        with _jobs_lock:
            j = _jobs.get(job_id)
            if j:
                j.review_applied = True
                j.message = (
                    f"Revisión aplicada · ok={summary['correct']} "
                    f"err={summary['incorrect']} skip={summary['skipped']}"
                )

        return {**summary, "train_started": train_started}

    @app.get("/api/jobs/{job_id}/preview")
    def job_preview(job_id: str) -> FileResponse:
        path = JOB_DIR / job_id / "preview_crossings.mp4"
        if not path.exists():
            raise HTTPException(404, "Preview no disponible")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/jobs/{job_id}/csv")
    def job_csv(job_id: str) -> FileResponse:
        path = JOB_DIR / job_id / "crossings.csv"
        if not path.exists():
            raise HTTPException(404, "CSV no disponible")
        return FileResponse(path, media_type="text/csv", filename="crossings.csv")

    # ----- Entrenamiento -----

    @app.post("/api/train/start")
    def train_start(req: TrainStartRequest) -> dict[str, Any]:
        path = _resolve_video(req.video_id)
        if req.kind not in {"detect", "recognize"}:
            raise HTTPException(400, "kind debe ser detect o recognize")
        # Un solo id: runtime y sesión en disco comparten el mismo id
        session_id = uuid.uuid4().hex[:12]
        save_session(
            ROOT,
            TrainSession(
                id=session_id,
                kind=req.kind,
                video_path=str(path),
                status="scanning",
                message="Iniciando escaneo…",
            ),
        )
        with _train_lock:
            _train_runtime[session_id] = {
                "status": "scanning",
                "message": "Iniciando escaneo…",
                "frame": 0,
                "total": 0,
                "candidates": 0,
                "cancel": False,
                "real_id": session_id,
            }

        thread = threading.Thread(
            target=_run_train_scan,
            args=(session_id, path, req),
            daemon=True,
        )
        thread.start()
        return {"session_id": session_id}

    # Rutas estáticas ANTES de /api/train/{session_id} para que FastAPI
    # no interprete "sessions" / "export" como un id de sesión.
    @app.get("/api/train/sessions")
    def train_sessions() -> dict[str, Any]:
        root = training_root(ROOT) / "sessions"
        items = []
        if root.exists():
            for p in sorted(root.glob("*/session.json"), reverse=True):
                data = json.loads(p.read_text(encoding="utf-8"))
                pending = sum(
                    1 for c in data.get("candidates", []) if c.get("status") == "pending"
                )
                items.append(
                    {
                        "id": data["id"],
                        "kind": data.get("kind"),
                        "status": data.get("status"),
                        "candidates": len(data.get("candidates", [])),
                        "pending": pending,
                        "message": data.get("message", ""),
                    }
                )
        return {"sessions": items}

    @app.post("/api/train/export")
    def train_export() -> dict[str, Any]:
        detect = export_detect_dataset(ROOT)
        recognize = export_recognize_dataset(ROOT)
        return {**detect, "recognize_labeled": recognize.get("labeled", 0)}

    # ----- Modelos -----

    @app.get("/api/models")
    def api_list_models(kind: str | None = None) -> dict[str, Any]:
        items = list_models(ROOT, kind=kind)
        return {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "kind": m.kind,
                    "status": m.status,
                    "message": m.message,
                    "parent_id": m.parent_id,
                    "metrics": m.metrics,
                    "updated_at": m.updated_at,
                    "ready": m.status == "ready",
                }
                for m in items
            ]
        }

    @app.post("/api/models")
    def api_create_model(req: ModelCreateRequest) -> dict[str, Any]:
        try:
            info = create_model(
                ROOT,
                name=req.name,
                kind=req.kind,
                parent_id=req.parent_id,
                notes=req.notes,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return asdict(info)

    @app.post("/api/models/{model_id}/train")
    def api_train_model(model_id: str, req: ModelTrainRequest) -> dict[str, Any]:
        info = get_model(ROOT, model_id)
        if info is None:
            raise HTTPException(404, "Modelo no encontrado")
        with _model_lock:
            # Limpiar runtimes huérfanos (hilo terminó pero quedó "training")
            for mid, r in list(_model_runtime.items()):
                if r.get("status") != "training":
                    continue
                other = get_model(ROOT, mid)
                if other and other.status in {"ready", "error", "empty"}:
                    r["status"] = other.status
                    r["message"] = other.message or r.get("message", "")
            if any(r.get("status") == "training" for r in _model_runtime.values()):
                raise HTTPException(409, "Ya hay un entrenamiento en curso")
            _model_runtime[model_id] = {
                "status": "training",
                "message": (
                    "Exportando etiquetas de reconocimiento…"
                    if info.kind == "recognize"
                    else "Iniciando entrenamiento YOLO…"
                ),
                "model_id": model_id,
                "kind": info.kind,
            }

        thread = threading.Thread(
            target=_run_model_train,
            args=(model_id, req),
            daemon=True,
        )
        thread.start()
        return {"ok": True, "model_id": model_id, "kind": info.kind}

    @app.get("/api/models/{model_id}/train-status")
    def api_train_model_status(model_id: str) -> dict[str, Any]:
        with _model_lock:
            rt = dict(_model_runtime.get(model_id) or {})
        info = get_model(ROOT, model_id)
        if info is None and not rt:
            raise HTTPException(404, "Modelo no encontrado")
        status = rt.get("status") or (info.status if info else "unknown")
        message = rt.get("message") or (info.message if info else "")
        return {
            "model_id": model_id,
            "kind": (info.kind if info else rt.get("kind")),
            "status": status,
            "message": message,
            "metrics": (info.metrics if info else {}),
        }

    @app.post("/api/train/{session_id}/stop")
    def train_stop(session_id: str) -> dict[str, Any]:
        with _train_lock:
            rt = _train_runtime.get(session_id)
            if not rt:
                # puede ser el id real de sesión ya persistida
                try:
                    sess = load_session(ROOT, session_id)
                    sess.cancel_requested = True
                    save_session(ROOT, sess)
                except FileNotFoundError as exc:
                    raise HTTPException(404, "Sesión no encontrada") from exc
                return {"ok": True}
            rt["cancel"] = True
            rt["message"] = "Deteniendo escaneo…"
        return {"ok": True}

    @app.get("/api/train/{session_id}")
    def train_status(session_id: str) -> dict[str, Any]:
        with _train_lock:
            rt = _train_runtime.get(session_id)
            if rt is None:
                # por si el cliente consulta con otro alias
                for _rid, candidate in _train_runtime.items():
                    if candidate.get("real_id") == session_id:
                        rt = candidate
                        break
            runtime = dict(rt) if rt else None

        try:
            sess = load_session(ROOT, session_id)
        except FileNotFoundError:
            if runtime:
                return {
                    "session_id": session_id,
                    "real_id": runtime.get("real_id") or session_id,
                    "status": runtime.get("status", "scanning"),
                    "message": runtime.get("message", ""),
                    "frame": runtime.get("frame", 0),
                    "total": runtime.get("total", 0),
                    "candidates": runtime.get("candidates", 0),
                    "reviewed": 0,
                    "pending": 0,
                    "cursor": 0,
                    "kind": None,
                }
            raise HTTPException(404, "Sesión no encontrada") from None

        pending = sum(1 for c in sess.candidates if c.status == "pending")
        reviewed = len(sess.candidates) - pending
        return {
            "session_id": sess.id,
            "real_id": sess.id,
            "kind": sess.kind,
            "status": sess.status,
            "message": sess.message,
            "frame": (runtime or {}).get("frame", 0),
            "total": (runtime or {}).get("total", 0),
            "candidates": len(sess.candidates),
            "pending": pending,
            "reviewed": reviewed,
            "cursor": sess.cursor,
        }

    @app.get("/api/train/{session_id}/next")
    def train_next(session_id: str) -> dict[str, Any]:
        try:
            sess = load_session(ROOT, session_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Sesión no encontrada") from exc

        # avanzar cursor al próximo pending
        while sess.cursor < len(sess.candidates):
            c = sess.candidates[sess.cursor]
            if c.status == "pending":
                break
            sess.cursor += 1
        save_session(ROOT, sess)

        if sess.cursor >= len(sess.candidates):
            sess.status = "done"
            sess.message = "Revisión completa"
            save_session(ROOT, sess)
            return {"done": True, "message": sess.message, "can_prev": len(sess.candidates) > 0}

        return _candidate_payload(session_id, sess)

    @app.post("/api/train/{session_id}/prev")
    def train_prev(session_id: str) -> dict[str, Any]:
        """Vuelve al candidato anterior y lo deja pendiente para corregir."""
        try:
            sess = load_session(ROOT, session_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Sesión no encontrada") from exc

        if not sess.candidates:
            raise HTTPException(400, "No hay candidatos")

        # Si estamos al final (done) o en un pending, el índice actual es cursor
        cur = min(sess.cursor, len(sess.candidates))
        # Si el actual es pending y aún no etiquetado, retroceder uno;
        # si cursor apunta past end, ir al último.
        if cur >= len(sess.candidates):
            prev_idx = len(sess.candidates) - 1
        elif sess.candidates[cur].status == "pending" and cur > 0:
            prev_idx = cur - 1
        elif sess.candidates[cur].status != "pending":
            # cursor quedó en uno ya revisado (raro); reabrir ese
            prev_idx = cur
        else:
            raise HTTPException(400, "Ya estás en el primer candidato")

        target = sess.candidates[prev_idx]
        target.status = "pending"
        target.reviewed_at = None
        # en recognize conservar sugerencia OCR si había
        sess.cursor = prev_idx
        sess.status = "review"
        sess.message = "Revisión en curso"
        save_session(ROOT, sess)
        return _candidate_payload(session_id, sess)

    @app.post("/api/train/{session_id}/label")
    def train_label(session_id: str, req: TrainLabelRequest) -> dict[str, Any]:
        try:
            sess = load_session(ROOT, session_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Sesión no encontrada") from exc

        cand = next((c for c in sess.candidates if c.id == req.candidate_id), None)
        if cand is None:
            raise HTTPException(404, "Candidato no encontrado")

        if sess.kind == "detect":
            if req.verdict not in {"yes", "no", "skip"}:
                raise HTTPException(400, "verdict: yes | no | skip")
            if req.verdict == "yes":
                cand.status = "positive"
            elif req.verdict == "no":
                cand.status = "negative"
            else:
                cand.status = "skipped"
        else:
            if req.verdict == "skip":
                cand.status = "skipped"
            elif req.bib and req.bib.strip():
                cand.status = "labeled"
                cand.label_bib = req.bib.strip()
            elif req.verdict == "yes" and cand.label_bib:
                cand.status = "labeled"
            else:
                raise HTTPException(400, "Indicá el número del dorsal o skip")

        cand.reviewed_at = time.time()
        # apuntar cursor al siguiente
        for i, c in enumerate(sess.candidates):
            if c.id == cand.id:
                sess.cursor = i + 1
                break
        pending = sum(1 for c in sess.candidates if c.status == "pending")
        if pending == 0:
            sess.status = "done"
            sess.message = "Revisión completa"
        save_session(ROOT, sess)
        return {"ok": True, "pending": pending, "status": sess.status}

    @app.get("/api/train/{session_id}/image/{cand_id}")
    def train_image(session_id: str, cand_id: str) -> FileResponse:
        path = session_dir(ROOT, session_id) / "images" / f"{Path(cand_id).name}.jpg"
        # id ya incluye nombre completo del archivo sin forzar
        alt = session_dir(ROOT, session_id) / "images" / Path(cand_id).name
        if not alt.suffix:
            alt = Path(str(alt) + ".jpg")
        # candidatos guardan image_rel = images/{cid}.jpg donde cid puede tener todo
        sess = load_session(ROOT, session_id)
        cand = next((c for c in sess.candidates if c.id == cand_id), None)
        if not cand:
            raise HTTPException(404, "Candidato no encontrado")
        img = session_dir(ROOT, session_id) / cand.image_rel
        if not img.exists():
            raise HTTPException(404, "Imagen no encontrada")
        return FileResponse(img, media_type="image/jpeg")

    @app.get("/api/train/{session_id}/context/{cand_id}")
    def train_context(session_id: str, cand_id: str) -> FileResponse:
        sess = load_session(ROOT, session_id)
        cand = next((c for c in sess.candidates if c.id == cand_id), None)
        if not cand or not cand.context_rel:
            raise HTTPException(404, "Contexto no encontrado")
        img = session_dir(ROOT, session_id) / cand.context_rel
        if not img.exists():
            raise HTTPException(404, "Contexto no encontrado")
        return FileResponse(img, media_type="image/jpeg")

    # ----- Generación de dorsales -----

    @app.post("/api/bibs/preview")
    def bib_preview(req: BibGenerateRequest) -> Response:
        try:
            style = _bib_style_from_req(req)
            png = bib_to_png_bytes(req.number.strip() or "0", style)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return Response(content=png, media_type="image/png")

    @app.post("/api/bibs/generate")
    def bib_generate(req: BibGenerateRequest) -> Response:
        try:
            style = _bib_style_from_req(req)
            if req.start is not None and req.end is not None:
                numbers = expand_number_range(req.start, req.end, req.pad_width)
            else:
                numbers = [req.number.strip() or "0"]
            if len(numbers) == 1:
                png = bib_to_png_bytes(numbers[0], style)
                return Response(
                    content=png,
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f'attachment; filename="dorsal_{numbers[0]}.png"'
                    },
                )
            zdata = generate_bib_zip(numbers, style)
            return Response(
                content=zdata,
                media_type="application/zip",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="dorsales_{numbers[0]}_{numbers[-1]}.zip"'
                    )
                },
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


def _candidate_payload(session_id: str, sess: Any) -> dict[str, Any]:
    c = sess.candidates[sess.cursor]
    pending = sum(1 for x in sess.candidates if x.status == "pending")
    return {
        "done": False,
        "index": sess.cursor,
        "total": len(sess.candidates),
        "pending": pending,
        "can_prev": sess.cursor > 0,
        "candidate": {
            "id": c.id,
            "frame": c.frame,
            "track_id": c.track_id,
            "score": c.score,
            "method": c.method,
            "kind": c.kind,
            "label_bib": c.label_bib,
            "image_url": f"/api/train/{session_id}/image/{c.id}",
            "context_url": (
                f"/api/train/{session_id}/context/{c.id}" if c.context_rel else None
            ),
        },
    }


def _resolve_video(video_id: str) -> Path:
    if video_id.startswith("data:"):
        path = ROOT / "data" / video_id.removeprefix("data:")
    else:
        # evitar path traversal
        name = Path(video_id).name
        path = UPLOAD_DIR / name
        if not path.exists():
            # permitir también data/*.mp4 por nombre
            alt = ROOT / "data" / name
            if alt.exists():
                path = alt
    if not path.exists() or not path.is_file():
        raise HTTPException(404, f"Video no encontrado: {video_id}")
    return path.resolve()


def _run_job(job_id: str, req: ProcessRequest) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = JobStatus.running
        job.message = "Procesando video…"

    def on_progress(payload: dict[str, Any]) -> None:
        with _jobs_lock:
            j = _jobs[job_id]
            j.message = str(payload.get("message") or j.message)
            j.frame = int(payload.get("frame") or 0)
            j.total_frames = int(payload.get("total") or 0)
            j.live_seq = int(payload.get("live_seq") or j.live_seq)
            if "events" in payload:
                j.events = list(payload["events"])
            snap = payload.get("new_snapshot")
            if snap and snap not in j.snapshots:
                j.snapshots.append(str(snap))
            if "pending" in payload:
                j.pending = list(payload["pending"])

    def should_cancel() -> bool:
        with _jobs_lock:
            return _jobs[job_id].cancel_requested

    try:
        finish = FinishLine(
            Point(req.x0, req.y0),
            Point(req.x1, req.y1),
        )
        zone = None
        if None not in (req.zx0, req.zy0, req.zx1, req.zy1):
            zone = DetectionZone(
                float(req.zx0), float(req.zy0), float(req.zx1), float(req.zy1)
            ).normalized()
        info = probe_device(prefer_gpu=not req.cpu)
        detect_id = req.detect_model_id
        # Si eligió reconocimiento con detector asociado y no hay detector manual
        if not detect_id and req.recognize_model_id:
            rec = get_model(ROOT, req.recognize_model_id)
            if rec and rec.metrics.get("detect_model_id"):
                detect_id = str(rec.metrics["detect_model_id"])
        known_bibs = load_recognize_known_bibs(ROOT, req.recognize_model_id)
        events = process_video(
            Path(job.video_path),
            finish,
            Path(job.output_dir),
            conf=req.conf,
            max_frames=req.max_frames,
            device_info=info,
            save_preview=True,
            enable_ocr=req.enable_ocr,
            bib_model=resolve_weights(ROOT, detect_id),
            known_bibs=known_bibs or None,
            detection_zone=zone,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        with _jobs_lock:
            cancelled = job.cancel_requested
            job.status = JobStatus.cancelled if cancelled else JobStatus.done
            job.message = (
                f"Detenido · {len(events)} cruces parciales"
                if cancelled
                else f"{len(events)} cruces detectados"
            )
            job.events = [asdict(e) for e in events]
            job.preview_url = f"/api/jobs/{job_id}/preview"
            job.csv_url = f"/api/jobs/{job_id}/csv"
            job.finished_at = time.time()
            for ev in events:
                if ev.snapshot:
                    name = Path(ev.snapshot).name
                    if name not in job.snapshots:
                        job.snapshots.append(name)
    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            job.status = JobStatus.error
            job.message = str(exc)
            job.finished_at = time.time()


def _run_train_scan(runtime_id: str, path: Path, req: TrainStartRequest) -> None:
    def on_progress(payload: dict[str, Any]) -> None:
        with _train_lock:
            rt = _train_runtime.get(runtime_id)
            if not rt:
                return
            rt["message"] = payload.get("message", rt.get("message", ""))
            rt["frame"] = payload.get("frame", 0)
            rt["total"] = payload.get("total", 0)
            rt["candidates"] = payload.get("candidates", 0)
            if payload.get("session_id"):
                rt["real_id"] = payload["session_id"]
            rt["status"] = payload.get("status", rt.get("status", "scanning"))

    def should_cancel() -> bool:
        with _train_lock:
            rt = _train_runtime.get(runtime_id)
            return bool(rt and rt.get("cancel"))

    try:
        scan_fn = (
            scan_detect_candidates if req.kind == "detect" else scan_recognize_candidates
        )
        bib_w = resolve_weights(ROOT, req.detect_model_id)
        session = scan_fn(
            path,
            ROOT,
            conf=req.conf,
            max_frames=req.max_frames,
            frame_stride=req.frame_stride,
            max_candidates=req.max_candidates,
            min_score=req.min_score,
            cpu=req.cpu,
            bib_weights=bib_w,
            session_id=runtime_id,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        with _train_lock:
            rt = _train_runtime.get(runtime_id)
            if rt is not None:
                rt["real_id"] = session.id
                rt["status"] = session.status
                rt["message"] = session.message
                rt["candidates"] = len(session.candidates)
    except Exception as exc:  # noqa: BLE001
        with _train_lock:
            rt = _train_runtime.get(runtime_id)
            if rt is not None:
                rt["status"] = "error"
                rt["message"] = str(exc)


def _run_model_train(model_id: str, req: ModelTrainRequest) -> None:
    def on_progress(payload: dict[str, Any]) -> None:
        with _model_lock:
            rt = _model_runtime.get(model_id)
            if not rt:
                return
            rt["status"] = payload.get("status", rt.get("status"))
            rt["message"] = payload.get("message", rt.get("message", ""))

    try:
        info = get_model(ROOT, model_id)
        if info is None:
            raise FileNotFoundError(model_id)
        if info.kind == "detect":
            train_detect_model(
                ROOT,
                model_id,
                epochs=req.epochs,
                batch=req.batch,
                cpu=req.cpu,
                on_progress=on_progress,
            )
        else:
            register_recognize_model(
                ROOT,
                model_id,
                detect_model_id=req.detect_model_id,
                on_progress=on_progress,
            )
        with _model_lock:
            info2 = get_model(ROOT, model_id)
            _model_runtime[model_id] = {
                "status": info2.status if info2 else "ready",
                "message": info2.message if info2 else "Listo",
                "model_id": model_id,
                "kind": info2.kind if info2 else info.kind,
            }
    except Exception as exc:  # noqa: BLE001
        with _model_lock:
            _model_runtime[model_id] = {
                "status": "error",
                "message": str(exc),
                "model_id": model_id,
            }


app = create_app()
