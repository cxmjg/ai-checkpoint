"""Propuesta de ROI de dorsal: modelo YOLO bib o fallback HSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from bib_timing.training.bib_roi import BibProposal, find_bib_roi, is_landscape_bib

# Caché por (ruta, mtime, size): si se pisa weights.pt, se recarga.
_bib_models: dict[tuple[str, int, int], YOLO] = {}


def invalidate_bib_cache(path: Path | None = None) -> None:
    """Invalida el detector en memoria (todo o una ruta concreta)."""
    if path is None:
        _bib_models.clear()
        return
    key_prefix = str(path.resolve())
    for k in list(_bib_models):
        if k[0] == key_prefix:
            del _bib_models[k]


def _load_bib(path: Path) -> YOLO:
    resolved = path.resolve()
    st = resolved.stat()
    # mtime + size: más robusto si el SO reutiliza mtime
    key = (str(resolved), int(st.st_mtime_ns), int(st.st_size))
    for stale in [k for k in _bib_models if k[0] == key[0] and k != key]:
        del _bib_models[stale]
    if key not in _bib_models:
        _bib_models[key] = YOLO(str(resolved))
    return _bib_models[key]


def propose_bib_roi(
    person_bgr: np.ndarray,
    *,
    bib_weights: Path | None = None,
    conf: float = 0.25,
    device: str = "cpu",
) -> BibProposal | None:
    """
    Si hay pesos de detector bib, corre YOLO sobre el crop de persona.
    Solo acepta cajas horizontales (más anchas que altas). Si no, HSV o None.
    """
    if bib_weights is not None and bib_weights.exists() and person_bgr.size > 0:
        model = _load_bib(bib_weights)
        results = model.predict(
            person_bgr,
            conf=conf,
            imgsz=320,
            device=device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy()
            # probar cajas por confianza; quedarse con la mejor horizontal
            order = confs.argsort()[::-1]
            h, w = person_bgr.shape[:2]
            for i in order:
                x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                prop = BibProposal(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(confs[i]),
                    method="yolo_bib",
                )
                if is_landscape_bib(prop):
                    return prop
    return find_bib_roi(person_bgr)
