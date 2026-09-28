"""Dataset de entrenamiento (detección / reconocimiento de dorsal)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TrainKind(str, Enum):
    detect = "detect"
    recognize = "recognize"


@dataclass
class Candidate:
    id: str
    kind: str
    frame: int
    track_id: int | None
    image_rel: str
    context_rel: str | None
    # caja del dorsal relativa a la imagen de persona (YOLO-ready)
    box_xyxy: tuple[int, int, int, int] | None
    person_w: int
    person_h: int
    score: float
    method: str
    status: str = "pending"  # pending | positive | negative | labeled | skipped
    label_bib: str | None = None
    reviewed_at: float | None = None


@dataclass
class TrainSession:
    id: str
    kind: str
    video_path: str
    created_at: float = field(default_factory=time.time)
    status: str = "scanning"  # scanning | review | done | error
    message: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    cursor: int = 0
    cancel_requested: bool = False


def training_root(project_root: Path) -> Path:
    root = project_root / "data" / "training"
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_dir(project_root: Path, session_id: str) -> Path:
    d = training_root(project_root) / "sessions" / session_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(exist_ok=True)
    (d / "context").mkdir(exist_ok=True)
    return d


def save_session(project_root: Path, session: TrainSession) -> None:
    d = session_dir(project_root, session.id)
    payload = {
        "id": session.id,
        "kind": session.kind,
        "video_path": session.video_path,
        "created_at": session.created_at,
        "status": session.status,
        "message": session.message,
        "cursor": session.cursor,
        "candidates": [asdict(c) for c in session.candidates],
    }
    (d / "session.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_session(project_root: Path, session_id: str) -> TrainSession:
    path = session_dir(project_root, session_id) / "session.json"
    if not path.exists():
        raise FileNotFoundError(session_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    cands: list[Candidate] = []
    for c in data.get("candidates", []):
        box = c.get("box_xyxy")
        if isinstance(box, list) and len(box) == 4:
            c["box_xyxy"] = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        cands.append(Candidate(**c))
    return TrainSession(
        id=data["id"],
        kind=data["kind"],
        video_path=data["video_path"],
        created_at=data.get("created_at", time.time()),
        status=data.get("status", "review"),
        message=data.get("message", ""),
        candidates=cands,
        cursor=int(data.get("cursor", 0)),
    )


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def export_detect_dataset(project_root: Path) -> dict[str, Any]:
    """
    Exporta dataset YOLO clase bib.

    - Positivos: imagen de contexto + label con caja del dorsal.
    - Negativos: misma imagen de contexto (o crop) con label vacío → fondo /
      hard-negative para que YOLO no dispare ahí.
    """
    root = training_root(project_root)
    pos = root / "bib_detect" / "positive"
    neg = root / "bib_detect" / "negative"
    yolo_img = root / "bib_detect" / "yolo" / "images"
    yolo_lbl = root / "bib_detect" / "yolo" / "labels"
    for p in (pos, neg, yolo_img, yolo_lbl):
        p.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = n_neg_yolo = 0
    sessions = root / "sessions"
    if not sessions.exists():
        return {"positive": 0, "negative": 0, "yolo_negatives": 0}

    for sess_path in sessions.glob("*/session.json"):
        data = json.loads(sess_path.read_text(encoding="utf-8"))
        if data.get("kind") != "detect":
            continue
        sess_dir = sess_path.parent
        for c in data.get("candidates", []):
            src = sess_dir / c["image_rel"]
            if not src.exists():
                continue
            cid = c["id"]
            ctx_rel = c.get("context_rel")
            box = c.get("box_xyxy")
            pw, ph = c.get("person_w") or 0, c.get("person_h") or 0

            if c.get("status") == "positive":
                dest = pos / f"{cid}.jpg"
                dest.write_bytes(src.read_bytes())
                n_pos += 1
                if ctx_rel and box and pw > 0 and ph > 0:
                    ctx = sess_dir / ctx_rel
                    if ctx.exists():
                        img_out = yolo_img / f"{cid}.jpg"
                        img_out.write_bytes(ctx.read_bytes())
                        x1, y1, x2, y2 = box
                        cx = ((x1 + x2) / 2) / pw
                        cy = ((y1 + y2) / 2) / ph
                        bw = (x2 - x1) / pw
                        bh = (y2 - y1) / ph
                        (yolo_lbl / f"{cid}.txt").write_text(
                            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n",
                            encoding="utf-8",
                        )
            elif c.get("status") == "negative":
                dest = neg / f"{cid}.jpg"
                dest.write_bytes(src.read_bytes())
                n_neg += 1
                # Fondo YOLO: sin cajas (hard negative)
                yolo_src = None
                if ctx_rel:
                    ctx = sess_dir / ctx_rel
                    if ctx.exists():
                        yolo_src = ctx
                if yolo_src is None:
                    yolo_src = src
                (yolo_img / f"neg_{cid}.jpg").write_bytes(yolo_src.read_bytes())
                (yolo_lbl / f"neg_{cid}.txt").write_text("", encoding="utf-8")
                n_neg_yolo += 1

    yaml_path = root / "bib_detect" / "yolo" / "data.yaml"
    yaml_path.write_text(
        f"path: { (root / 'bib_detect' / 'yolo').resolve() }\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: bib\n",
        encoding="utf-8",
    )
    return {
        "positive": n_pos,
        "negative": n_neg,
        "yolo_negatives": n_neg_yolo,
        "yolo_dir": str(root / "bib_detect" / "yolo"),
        "positive_dir": str(pos),
        "negative_dir": str(neg),
    }



def export_recognize_dataset(project_root: Path) -> dict[str, Any]:
    """Copia crops de reconocimiento etiquetados (número de dorsal)."""
    root = training_root(project_root)
    out = root / "bib_recognize" / "labeled"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    manifest: list[dict[str, Any]] = []
    sessions = root / "sessions"
    if sessions.exists():
        for sess_path in sessions.glob("*/session.json"):
            data = json.loads(sess_path.read_text(encoding="utf-8"))
            if data.get("kind") != "recognize":
                continue
            sess_dir = sess_path.parent
            for c in data.get("candidates", []):
                if c.get("status") != "labeled":
                    continue
                bib = (c.get("label_bib") or "").strip()
                if not bib:
                    continue
                src = sess_dir / c["image_rel"]
                if not src.exists():
                    continue
                cid = c["id"]
                dest = out / f"{bib}__{cid}.jpg"
                dest.write_bytes(src.read_bytes())
                manifest.append(
                    {
                        "id": cid,
                        "bib": bib,
                        "file": dest.name,
                        "frame": c.get("frame"),
                        "session": data.get("id"),
                    }
                )
                n += 1
    man_path = root / "bib_recognize" / "manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "labeled": n,
        "recognize_dir": str(out),
        "manifest": str(man_path),
    }
