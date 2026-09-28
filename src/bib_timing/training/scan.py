"""Escaneo de video para generar candidatos de entrenamiento."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
from rich.console import Console
from ultralytics import YOLO

from bib_timing.device import probe_device
from bib_timing.ocr import body_crop, read_bib_scored
from bib_timing.training.bib_detector import propose_bib_roi
from bib_timing.training.bib_roi import crop_proposal, is_landscape_bib
from bib_timing.training.dataset import (
    Candidate,
    TrainSession,
    new_session_id,
    save_session,
    session_dir,
)
from bib_timing.training.rejects import is_previously_rejected, load_rejected_bibs
from bib_timing.video_io import open_video

console = Console()
PERSON_CLASS = 0
ProgressCb = Callable[[dict[str, Any]], None]
CancelCb = Callable[[], bool]


def scan_detect_candidates(
    video_path: Path,
    project_root: Path,
    *,
    conf: float = 0.35,
    imgsz: int = 640,
    max_frames: int | None = None,
    frame_stride: int = 5,
    min_score: float = 0.14,
    max_candidates: int = 400,
    cpu: bool = False,
    bib_weights: Path | None = None,
    session_id: str | None = None,
    on_progress: ProgressCb | None = None,
    should_cancel: CancelCb | None = None,
) -> TrainSession:
    """Personas → ROI dorsal (YOLO bib o HSV) → cola de revisión sí/no."""
    device = probe_device(prefer_gpu=not cpu)
    sid = session_id or new_session_id()
    sdir = session_dir(project_root, sid)
    session = TrainSession(
        id=sid,
        kind="detect",
        video_path=str(video_path),
        status="scanning",
        message="Escaneando candidatos de dorsal…",
    )
    save_session(project_root, session)

    model = YOLO("yolo11n.pt")
    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames:
        total = min(total, max_frames) if total else max_frames

    frame_idx = 0
    last_kept: dict[int, int] = {}
    # Con modelo bib entrenado, los scores son conf YOLO (0–1);
    # el umbral lo define el usuario (min_score).
    score_floor = float(min_score)
    rejected = load_rejected_bibs(project_root, video_path, kind="detect")
    skipped_rejected = 0

    def emit(msg: str) -> None:
        session.message = msg
        if on_progress:
            on_progress(
                {
                    "session_id": sid,
                    "status": session.status,
                    "message": msg,
                    "frame": frame_idx,
                    "total": total,
                    "candidates": len(session.candidates),
                }
            )

    mode = f"bib={bib_weights.name}" if bib_weights else "HSV"
    extra = f" · omitidos prev={len(rejected)}" if rejected else ""
    emit(f"Escaneando candidatos de dorsal ({mode}){extra}…")

    try:
        while True:
            if should_cancel and should_cancel():
                session.status = "review"
                session.message = "Escaneo detenido — podés revisar lo acumulado"
                break
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % max(1, frame_stride) != 0:
                frame_idx += 1
                continue

            results = model.track(
                frame,
                persist=True,
                conf=conf,
                imgsz=imgsz,
                classes=[PERSON_CLASS],
                tracker="bytetrack.yaml",
                device=device.device,
                verbose=False,
            )
            result = results[0]
            if result.boxes is None:
                frame_idx += 1
                continue

            has_id = result.boxes.id is not None
            xyxy = result.boxes.xyxy.cpu().numpy()
            ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if has_id
                else list(range(len(xyxy)))
            )

            for box, tid in zip(xyxy, ids):
                tid = int(tid)
                if tid in last_kept and frame_idx - last_kept[tid] < frame_stride * 3:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                person = body_crop(frame, (x1, y1, x2, y2), pad=0.0)
                if person.size == 0:
                    continue
                prop = propose_bib_roi(
                    person,
                    bib_weights=bib_weights,
                    conf=max(0.15, conf - 0.1),
                    device=device.device,
                )
                if prop is None or prop.score < score_floor:
                    continue
                if not is_landscape_bib(prop):
                    continue

                ph, pw = person.shape[:2]
                box_t = (prop.x1, prop.y1, prop.x2, prop.y2)
                if is_previously_rejected(
                    frame=frame_idx,
                    box_xyxy=box_t,
                    person_w=pw,
                    person_h=ph,
                    rejected=rejected,
                    frame_tol=max(8, frame_stride * 2),
                ):
                    skipped_rejected += 1
                    last_kept[tid] = frame_idx
                    continue

                cid = f"{sid}_{frame_idx:06d}_{tid}_{len(session.candidates)}"
                crop = crop_proposal(person, prop)
                # Contexto limpio (sin rectángulo dibujado) para YOLO;
                # el overlay solo se usa en UI si hace falta.
                context = person.copy()
                img_rel = f"images/{cid}.jpg"
                ctx_rel = f"context/{cid}.jpg"
                cv2.imwrite(str(sdir / img_rel), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(str(sdir / ctx_rel), context, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

                session.candidates.append(
                    Candidate(
                        id=cid,
                        kind="detect",
                        frame=frame_idx,
                        track_id=tid,
                        image_rel=img_rel,
                        context_rel=ctx_rel,
                        box_xyxy=box_t,
                        person_w=pw,
                        person_h=ph,
                        score=prop.score,
                        method=prop.method,
                    )
                )
                last_kept[tid] = frame_idx

                if len(session.candidates) >= max_candidates:
                    session.message = f"Límite de {max_candidates} candidatos"
                    session.status = "review"
                    emit(session.message)
                    save_session(project_root, session)
                    return session

            if frame_idx % 30 == 0:
                emit(
                    f"Escaneando frame {frame_idx}"
                    + (f"/{total}" if total else "")
                    + f" · candidatos={len(session.candidates)}"
                )
                save_session(project_root, session)

            frame_idx += 1
    finally:
        cap.release()

    session.status = "review"
    session.message = (
        f"Listo: {len(session.candidates)} candidatos para revisar"
        + (f" · {skipped_rejected} ya rechazados omitidos" if skipped_rejected else "")
    )
    save_session(project_root, session)
    emit(session.message)
    console.print(f"[green]Sesión {sid}[/green] {session.message}")
    return session


def scan_recognize_candidates(
    video_path: Path,
    project_root: Path,
    *,
    conf: float = 0.35,
    imgsz: int = 640,
    max_frames: int | None = None,
    frame_stride: int = 8,
    max_candidates: int = 300,
    min_score: float = 0.14,
    cpu: bool = False,
    bib_weights: Path | None = None,
    session_id: str | None = None,
    on_progress: ProgressCb | None = None,
    should_cancel: CancelCb | None = None,
) -> TrainSession:
    """Propone crops de dorsal (idealmente con modelo detect); usuario confirma número."""
    device = probe_device(prefer_gpu=not cpu)
    sid = session_id or new_session_id()
    sdir = session_dir(project_root, sid)
    session = TrainSession(
        id=sid,
        kind="recognize",
        video_path=str(video_path),
        status="scanning",
        message="Escaneando dorsales para reconocimiento…",
    )
    save_session(project_root, session)

    model = YOLO("yolo11n.pt")
    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    ocr_gpu = device.device.startswith("cuda")
    frame_idx = 0
    last_kept: dict[int, int] = {}
    rejected_detect = load_rejected_bibs(project_root, video_path, kind="detect")

    def emit(msg: str) -> None:
        session.message = msg
        if on_progress:
            on_progress(
                {
                    "session_id": sid,
                    "status": session.status,
                    "message": msg,
                    "frame": frame_idx,
                    "total": total,
                    "candidates": len(session.candidates),
                }
            )

    mode = f"bib={bib_weights.name}" if bib_weights else "HSV"
    emit(f"Escaneando dorsales para reconocimiento ({mode})…")

    try:
        while True:
            if should_cancel and should_cancel():
                session.status = "review"
                session.message = "Escaneo detenido"
                break
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % max(1, frame_stride) != 0:
                frame_idx += 1
                continue

            results = model.track(
                frame,
                persist=True,
                conf=conf,
                imgsz=imgsz,
                classes=[PERSON_CLASS],
                tracker="bytetrack.yaml",
                device=device.device,
                verbose=False,
            )
            result = results[0]
            if result.boxes is None or result.boxes.id is None:
                frame_idx += 1
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(xyxy, ids):
                tid = int(tid)
                if tid in last_kept and frame_idx - last_kept[tid] < frame_stride * 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                person = body_crop(frame, (x1, y1, x2, y2), pad=0.0)
                prop = propose_bib_roi(
                    person,
                    bib_weights=bib_weights,
                    conf=max(0.15, conf - 0.1),
                    device=device.device,
                )
                if prop is None:
                    continue
                if prop.score < min_score:
                    continue
                if not is_landscape_bib(prop):
                    continue
                ph, pw = person.shape[:2]
                box_t = (prop.x1, prop.y1, prop.x2, prop.y2)
                # Reutilizar rechazos de detección del mismo video
                if is_previously_rejected(
                    frame=frame_idx,
                    box_xyxy=box_t,
                    person_w=pw,
                    person_h=ph,
                    rejected=rejected_detect,
                    frame_tol=max(8, frame_stride * 2),
                ):
                    last_kept[tid] = frame_idx
                    continue
                crop = crop_proposal(person, prop)
                if crop.size == 0:
                    continue
                suggested = None
                try:
                    suggested, _ = read_bib_scored(crop, use_gpu=ocr_gpu)
                except Exception:  # noqa: BLE001
                    suggested = None

                cid = f"{sid}_{frame_idx:06d}_{tid}_{len(session.candidates)}"
                img_rel = f"images/{cid}.jpg"
                ctx_rel = f"context/{cid}.jpg"
                cv2.imwrite(str(sdir / img_rel), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(
                    str(sdir / ctx_rel),
                    person.copy(),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )
                session.candidates.append(
                    Candidate(
                        id=cid,
                        kind="recognize",
                        frame=frame_idx,
                        track_id=tid,
                        image_rel=img_rel,
                        context_rel=ctx_rel,
                        box_xyxy=box_t,
                        person_w=pw,
                        person_h=ph,
                        score=prop.score,
                        method=prop.method,
                        label_bib=suggested,
                    )
                )
                last_kept[tid] = frame_idx
                if len(session.candidates) >= max_candidates:
                    session.status = "review"
                    session.message = f"Límite {max_candidates} candidatos"
                    save_session(project_root, session)
                    emit(session.message)
                    return session

            if frame_idx % 30 == 0:
                emit(
                    f"Escaneando {frame_idx}"
                    + (f"/{total}" if total else "")
                    + f" · {len(session.candidates)} candidatos"
                )
                save_session(project_root, session)
            frame_idx += 1
    finally:
        cap.release()

    session.status = "review"
    session.message = f"Listo: {len(session.candidates)} para etiquetar número"
    save_session(project_root, session)
    emit(session.message)
    return session
