"""Filtro de propuestas ya rechazadas en sesiones anteriores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bib_timing.training.dataset import training_root


def _video_key(video_path: str | Path) -> str:
    p = Path(video_path).resolve()
    return p.name.lower()


def _norm_box(
    box: tuple[int, int, int, int] | list[int], pw: int, ph: int
) -> tuple[float, float, float, float] | None:
    if pw <= 0 or ph <= 0:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    return (x1 / pw, y1 / ph, x2 / pw, y2 / ph)


def _iou_norm(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_rejected_bibs(
    project_root: Path, video_path: Path, *, kind: str = "detect"
) -> list[dict[str, Any]]:
    """Negativos previos del mismo video (para no volver a proponerlos)."""
    root = training_root(project_root) / "sessions"
    if not root.exists():
        return []
    want = _video_key(video_path)
    out: list[dict[str, Any]] = []
    for sess_path in root.glob("*/session.json"):
        data = json.loads(sess_path.read_text(encoding="utf-8"))
        if data.get("kind") != kind:
            continue
        if _video_key(data.get("video_path", "")) != want:
            continue
        for c in data.get("candidates", []):
            if c.get("status") != "negative":
                continue
            box = c.get("box_xyxy")
            pw, ph = int(c.get("person_w") or 0), int(c.get("person_h") or 0)
            if not box or len(box) != 4 or pw <= 0 or ph <= 0:
                continue
            nb = _norm_box(box, pw, ph)
            if nb is None:
                continue
            out.append(
                {
                    "frame": int(c.get("frame") or 0),
                    "track_id": c.get("track_id"),
                    "box_norm": nb,
                }
            )
    return out


def is_previously_rejected(
    *,
    frame: int,
    box_xyxy: tuple[int, int, int, int],
    person_w: int,
    person_h: int,
    rejected: list[dict[str, Any]],
    frame_tol: int = 10,
    min_iou: float = 0.35,
) -> bool:
    """True si ya se marcó como no-dorsal un recorte similar en este video."""
    if not rejected:
        return False
    cur = _norm_box(box_xyxy, person_w, person_h)
    if cur is None:
        return False
    for r in rejected:
        if abs(int(r["frame"]) - frame) > frame_tol:
            continue
        if _iou_norm(cur, r["box_norm"]) >= min_iou:
            return True
    return False
