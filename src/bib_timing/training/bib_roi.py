"""Propuesta de ROI de dorsal por color (celeste/blanco) dentro de un crop de persona."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Dorsal típico: más ancho que alto (horizontal). Si no cumple, se descarta.
MIN_BIB_ASPECT = 1.15  # width / height
MAX_BIB_ASPECT = 3.2


@dataclass
class BibProposal:
    """Caja del dorsal en coordenadas del crop de persona (píxeles)."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    method: str = "hsv_cyan_white"

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)


def is_landscape_bib(prop: BibProposal) -> bool:
    """True si el recorte es rectangular horizontal (solo dorsal)."""
    if prop.width < 16 or prop.height < 10:
        return False
    return MIN_BIB_ASPECT <= prop.aspect <= MAX_BIB_ASPECT


def find_bib_roi(person_bgr: np.ndarray) -> BibProposal | None:
    """
    Busca un rectángulo tipo dorsal: borde/área celeste + interior blanco.
    Solo acepta cajas más anchas que altas (formato horizontal).
    """
    if person_bgr is None or person_bgr.size == 0:
        return None
    h, w = person_bgr.shape[:2]
    if h < 40 or w < 30:
        return None

    hsv = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2HSV)

    # Celeste / cian (ajustable por carrera)
    cyan = cv2.inRange(hsv, (80, 40, 60), (110, 255, 255))
    # Blanco / casi blanco
    white = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

    # Zona probable del torso (evitar cabeza/piernas)
    torso = np.zeros_like(white)
    y0, y1 = int(0.18 * h), int(0.72 * h)
    x0, x1 = int(0.12 * w), int(0.88 * w)
    torso[y0:y1, x0:x1] = 255

    cyan = cv2.bitwise_and(cyan, torso)
    white = cv2.bitwise_and(white, torso)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)
    cyan = cv2.morphologyEx(cyan, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Combinar: blanco reforzado por cercanía a celeste
    cyan_dil = cv2.dilate(cyan, kernel, iterations=2)
    combined = cv2.bitwise_or(white, cv2.bitwise_and(white, cyan_dil))
    if cv2.countNonZero(cyan) > 30:
        combined = cv2.bitwise_or(combined, cyan_dil)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: BibProposal | None = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * w * h or area > 0.45 * w * h:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 12 or bh < 12:
            continue
        aspect = bw / max(1, bh)
        # Solo dorsales horizontales (más anchos que altos)
        if aspect < MIN_BIB_ASPECT or aspect > MAX_BIB_ASPECT:
            continue

        roi_white = white[y : y + bh, x : x + bw]
        roi_cyan = cyan[y : y + bh, x : x + bw]
        white_ratio = cv2.countNonZero(roi_white) / max(1, bw * bh)
        cyan_ratio = cv2.countNonZero(roi_cyan) / max(1, bw * bh)
        if white_ratio < 0.18 and cyan_ratio < 0.04:
            continue

        # Preferir torso medio
        cy = y + bh / 2
        center_bonus = 1.0 - abs(cy - 0.42 * h) / h
        score = white_ratio * 0.55 + cyan_ratio * 0.35 + max(0.0, center_bonus) * 0.2
        score *= min(1.0, area / (0.05 * w * h))

        pad = int(0.06 * max(bw, bh))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        cand = BibProposal(x1, y1, x2, y2, float(score))
        if not is_landscape_bib(cand):
            continue
        if best is None or cand.score > best.score:
            best = cand

    if best is None or best.score < 0.12:
        return None
    return best


def draw_proposal(person_bgr: np.ndarray, prop: BibProposal) -> np.ndarray:
    out = person_bgr.copy()
    cv2.rectangle(out, (prop.x1, prop.y1), (prop.x2, prop.y2), (0, 220, 255), 2)
    return out


def crop_proposal(person_bgr: np.ndarray, prop: BibProposal) -> np.ndarray:
    return person_bgr[prop.y1 : prop.y2, prop.x1 : prop.x2].copy()
