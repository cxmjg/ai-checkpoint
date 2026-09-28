"""Geometría de línea de meta, zona de detección y cruces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class FinishLine:
    """Segmento en coordenadas de píxel de la imagen."""

    a: Point
    b: Point

    @classmethod
    def from_tuple(
        cls, coords: tuple[tuple[float, float], tuple[float, float]]
    ) -> FinishLine:
        (x0, y0), (x1, y1) = coords
        return cls(Point(x0, y0), Point(x1, y1))


@dataclass(frozen=True)
class DetectionZone:
    """Rectángulo axis-aligned: área donde se hace OCR y recortes de dorsal."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_corners(cls, a: Point, b: Point) -> DetectionZone:
        return cls(
            min(a.x, b.x),
            min(a.y, b.y),
            max(a.x, b.x),
            max(a.y, b.y),
        )

    @classmethod
    def full_frame(cls, width: int, height: int, inset: float = 0.0) -> DetectionZone:
        dx = width * inset
        dy = height * inset
        return cls(dx, dy, width - dx, height - dy)

    def normalized(self) -> DetectionZone:
        return DetectionZone(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
        )

    def contains(self, p: Point) -> bool:
        z = self.normalized()
        return z.x0 <= p.x <= z.x1 and z.y0 <= p.y <= z.y1

    def contains_box(
        self, box: tuple[float, float, float, float], *, min_overlap: float = 0.35
    ) -> bool:
        """True si el centroide del torso está dentro o hay solapamiento suficiente."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = y1 + 0.65 * (y2 - y1)
        if self.contains(Point(cx, cy)):
            return True
        z = self.normalized()
        ix0 = max(z.x0, x1)
        iy0 = max(z.y0, y1)
        ix1 = min(z.x1, x2)
        iy1 = min(z.y1, y2)
        if ix1 <= ix0 or iy1 <= iy0:
            return False
        inter = (ix1 - ix0) * (iy1 - iy0)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        return (inter / area) >= min_overlap


def side_of_line(p: Point, line: FinishLine) -> float:
    """Signo del producto cruzado: >0 un lado, <0 el otro, ~0 sobre la línea."""
    return (line.b.x - line.a.x) * (p.y - line.a.y) - (line.b.y - line.a.y) * (
        p.x - line.a.x
    )


def crossed_line(prev: Point, curr: Point, line: FinishLine) -> bool:
    """True si el segmento prev→curr cruza el segmento de meta."""
    s0 = side_of_line(prev, line)
    s1 = side_of_line(curr, line)
    if s0 == 0 or s1 == 0:
        return abs(s0) + abs(s1) < 1e-6
    if (s0 > 0) == (s1 > 0):
        return False

    min_x = min(line.a.x, line.b.x) - 40
    max_x = max(line.a.x, line.b.x) + 40
    min_y = min(line.a.y, line.b.y) - 40
    max_y = max(line.a.y, line.b.y) + 40
    mx = (prev.x + curr.x) / 2
    my = (prev.y + curr.y) / 2
    return min_x <= mx <= max_x and min_y <= my <= max_y
