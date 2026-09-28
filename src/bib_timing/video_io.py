"""Apertura de video y conversión vía FFmpeg del sistema."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
from rich.console import Console

console = Console()


class VideoOpenError(RuntimeError):
    """No se pudo abrir o decodificar el video."""


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        _hint_codec(path)
        raise VideoOpenError(f"No se pudo abrir: {path}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        _hint_codec(path)
        raise VideoOpenError(f"No se pudo decodificar frames: {path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def _hint_codec(path: Path) -> None:
    codec = probe_video_codec(path)
    if codec and codec.lower() in {"av1", "av01", "vp9", "vp09"}:
        console.print(
            f"[yellow]Codec detectado:[/yellow] {codec}. "
            "OpenCV (wheel) suele fallar con AV1/VP9."
        )
        console.print(
            "Convertí a H.264:\n"
            f"  [bold]python -m bib_timing.cli convert {path} -o data/clip.mp4 "
            "--seconds 90[/bold]"
        )
    elif codec:
        console.print(f"[yellow]Codec:[/yellow] {codec}")
    console.print(
        "También podés usar FFmpeg a mano:\n"
        "  ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -an output.mp4"
    )


def probe_video_codec(path: Path) -> str | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, OSError):
        return None


def convert_to_h264(
    src: Path,
    dst: Path,
    *,
    start_sec: float = 0.0,
    seconds: float | None = 90.0,
    crf: int = 23,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VideoOpenError("ffmpeg no está en el PATH")

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(src)]
    if seconds is not None:
        cmd += ["-t", str(seconds)]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    console.print(f"Convirtiendo → {dst} …")
    subprocess.check_call(cmd)
    console.print(f"[green]OK[/green] {dst}")
    return dst
