"""OCR de dorsales + votación multi-frame (EasyOCR opcional)."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from rich.console import Console

console = Console()

_DIGITS = re.compile(r"\d{1,6}")


@lru_cache(maxsize=1)
def _reader(gpu: bool):
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR no instalado. pip install -e '.[ocr]' o pip install easyocr"
        ) from exc
    console.print(f"[bold]Cargando EasyOCR[/bold] (gpu={gpu})…")
    return easyocr.Reader(["en"], gpu=gpu, verbose=False)


def body_crop(
    frame: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    *,
    pad: float = 0.02,
) -> np.ndarray:
    """Recorte de cuerpo entero (bbox persona) con un poco de padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx1 = int(max(0, x1 - pad * bw))
    cy1 = int(max(0, y1 - pad * bh))
    cx2 = int(min(w, x2 + pad * bw))
    cy2 = int(min(h, y2 + pad * bh))
    if cx2 <= cx1 or cy2 <= cy1:
        return frame[int(max(0, y1)) : int(min(h, y2)), int(max(0, x1)) : int(min(w, x2))]
    return frame[cy1:cy2, cx1:cx2]


@dataclass
class OcrHit:
    bib: str
    score: float
    frame: int


@dataclass
class BibVotes:
    """Acumula lecturas OCR por track y elige la más votada."""

    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    score_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    hits: list[OcrHit] = field(default_factory=list)
    best_hit: OcrHit | None = None

    def add(self, bib: str | None, score: float, frame: int) -> None:
        if not bib:
            return
        self.counts[bib] += 1
        self.score_sum[bib] += float(score)
        hit = OcrHit(bib=bib, score=float(score), frame=frame)
        self.hits.append(hit)
        if self.best_hit is None or hit.score > self.best_hit.score:
            self.best_hit = hit

    def winner(self) -> tuple[str | None, float, dict[str, int]]:
        """
        Gana el número con más apariciones.
        Empate → mayor suma de confianzas OCR.
        """
        if not self.counts:
            return None, 0.0, {}
        ranked = sorted(
            self.counts.keys(),
            key=lambda b: (self.counts[b], self.score_sum[b]),
            reverse=True,
        )
        bib = ranked[0]
        total = sum(self.counts.values())
        conf = self.counts[bib] / total if total else 0.0
        return bib, conf, dict(self.counts)


def read_bib_scored(
    crop_bgr: np.ndarray, *, use_gpu: bool = True
) -> tuple[str | None, float]:
    """Devuelve (número, confianza) del mejor candidato en el recorte."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    ch, cw = crop_bgr.shape[:2]
    if max(ch, cw) < 120:
        scale = 120 / max(ch, cw)
        crop_bgr = cv2.resize(
            crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    try:
        reader = _reader(use_gpu)
        results = reader.readtext(rgb, detail=1, paragraph=False)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]OCR falló:[/yellow] {exc}")
        return None, 0.0

    candidates: list[tuple[float, str]] = []
    for _bbox, text, score in results:
        digits = "".join(_DIGITS.findall(re.sub(r"\D", "", text)))
        if not digits:
            continue
        weight = float(score)
        if 2 <= len(digits) <= 5:
            weight += 0.15
        candidates.append((weight, digits))

    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda t: t[0], reverse=True)
    best_w, best_bib = candidates[0]
    # normalizar peso a ~0-1 para acumular
    return best_bib, min(1.0, best_w)


def refine_bib_with_known(
    bib: str | None,
    score: float,
    known_bibs: set[str] | None,
) -> tuple[str | None, float]:
    """
    Refuerza la lectura OCR con números del dataset de reconocimiento.
    - Si el OCR coincide con un conocido → sube confianza.
    - Si no coincide, busca el conocido más cercano (distancia de edición ≤ 1).
    """
    if not known_bibs:
        return bib, score
    if bib and bib in known_bibs:
        return bib, min(1.0, float(score) + 0.2)
    if bib:
        best: str | None = None
        best_d = 99
        for k in known_bibs:
            d = _edit_distance(bib, k)
            if d < best_d:
                best_d = d
                best = k
        if best is not None and best_d <= 1 and abs(len(bib) - len(best)) <= 1:
            return best, min(1.0, max(float(score), 0.55) + 0.1 * (1 - best_d))
    return bib, score


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def read_bib(crop_bgr: np.ndarray, *, use_gpu: bool = True) -> str | None:
    bib, _ = read_bib_scored(crop_bgr, use_gpu=use_gpu)
    return bib


# compat: alias antiguo
def torso_crop(
    frame: np.ndarray, box_xyxy: tuple[float, float, float, float]
) -> np.ndarray:
    return body_crop(frame, box_xyxy)
