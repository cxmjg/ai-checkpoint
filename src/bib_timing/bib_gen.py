"""Generación de dorsales imprimibles con marcadores tipo QR."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass
class BibStyle:
    bib_bg: str = "#FFFFFF"
    number_color: str = "#111111"
    marker_color: str = "#111111"
    border_color: str = "#111111"
    race_text: str = "Maratón 2026"
    race_color: str = "#444444"
    page_bg: str = "#F0F0F0"
    width: int = 900
    height: int = 1100
    bib_margin: int = 48


def _hex_rgb(color: str) -> tuple[int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"Color inválido: {color}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_finder(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
    bg: tuple[int, int, int],
) -> None:
    """Patrón de esquina estilo QR (3 anillos concéntricos)."""
    outer = size
    mid = int(size * 5 / 7)
    inner = int(size * 3 / 7)
    ox = (outer - mid) // 2
    ix = (outer - inner) // 2
    draw.rectangle([x, y, x + outer - 1, y + outer - 1], fill=color)
    draw.rectangle(
        [x + ox, y + ox, x + ox + mid - 1, y + ox + mid - 1], fill=bg
    )
    draw.rectangle(
        [x + ix, y + ix, x + ix + inner - 1, y + ix + inner - 1], fill=color
    )


def _draw_alignment(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    size: int,
    color: tuple[int, int, int],
    bg: tuple[int, int, int],
) -> None:
    """Marcador de alineación (esquina inferior derecha, distinto a los finders)."""
    half = size // 2
    draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=color)
    draw.ellipse(
        [cx - half // 2, cy - half // 2, cx + half // 2, cy + half // 2],
        fill=bg,
    )
    draw.ellipse(
        [cx - half // 4, cy - half // 4, cx + half // 4, cy + half // 4],
        fill=color,
    )


def render_bib(number: str | int, style: BibStyle | None = None) -> Image.Image:
    """
    Página con dorsal + texto de carrera debajo (fuera del área del dorsal).
    El dorsal incluye finders en 3 esquinas y un marcador de orientación.
    """
    style = style or BibStyle()
    bib_bg = _hex_rgb(style.bib_bg)
    number_color = _hex_rgb(style.number_color)
    marker = _hex_rgb(style.marker_color)
    border = _hex_rgb(style.border_color)
    race_color = _hex_rgb(style.race_color)
    page_bg = _hex_rgb(style.page_bg)

    w, h = style.width, style.height
    margin = style.bib_margin
    race_band = max(72, h // 10)
    bib_h = h - race_band - margin
    bib_w = w - 2 * margin

    img = Image.new("RGB", (w, h), page_bg)
    draw = ImageDraw.Draw(img)

    # Placa del dorsal
    x0, y0 = margin, margin // 2
    x1, y1 = margin + bib_w - 1, y0 + bib_h - 1
    draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=bib_bg, outline=border, width=6)

    finder = max(56, bib_w // 8)
    inset = max(18, bib_w // 28)
    _draw_finder(draw, x0 + inset, y0 + inset, finder, marker, bib_bg)
    _draw_finder(draw, x1 - inset - finder + 1, y0 + inset, finder, marker, bib_bg)
    _draw_finder(draw, x0 + inset, y1 - inset - finder + 1, finder, marker, bib_bg)
    align = max(28, finder // 2)
    _draw_alignment(
        draw,
        x1 - inset - align // 2,
        y1 - inset - align // 2,
        align,
        marker,
        bib_bg,
    )

    # Número centrado
    text = str(number).strip()
    max_tw = bib_w - 2 * (inset + finder + 8)
    max_th = bib_h - 2 * (inset + finder // 2)
    font_size = min(int(bib_h * 0.42), 320)
    font = _font(font_size, bold=True)
    while font_size > 40:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= max_tw and th <= max_th:
            break
        font_size -= 8
        font = _font(font_size, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x0 + (bib_w - tw) // 2 - bbox[0]
    ty = y0 + (bib_h - th) // 2 - bbox[1]
    draw.text((tx, ty), text, fill=number_color, font=font)

    # Texto de carrera (fuera del dorsal)
    race = (style.race_text or "").strip()
    if race:
        rfont = _font(max(22, w // 28), bold=False)
        rb = draw.textbbox((0, 0), race, font=rfont)
        rw, rh = rb[2] - rb[0], rb[3] - rb[1]
        rx = (w - rw) // 2 - rb[0]
        ry = y1 + (race_band - rh) // 2 - rb[1] + 4
        draw.text((rx, ry), race, fill=race_color, font=rfont)

    return img


def bib_to_png_bytes(number: str | int, style: BibStyle | None = None) -> bytes:
    img = render_bib(number, style)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_bib_zip(
    numbers: list[str],
    style: BibStyle,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for n in numbers:
            name = f"dorsal_{n}.png"
            zf.writestr(name, bib_to_png_bytes(n, style))
    return buf.getvalue()


def expand_number_range(start: int, end: int, width: int = 0) -> list[str]:
    if end < start:
        raise ValueError("El número final debe ser ≥ al inicial")
    if end - start > 5000:
        raise ValueError("Máximo 5000 dorsales por lote")
    out: list[str] = []
    for n in range(start, end + 1):
        s = str(n)
        if width > 0:
            s = s.zfill(width)
        out.append(s)
    return out
