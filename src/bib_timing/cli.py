"""CLI de bib-timing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from bib_timing.device import probe_device
from bib_timing.finish_line import FinishLine, Point
from bib_timing.pipeline import default_horizontal_line, process_video
from bib_timing.video_io import convert_to_h264, open_video

app = typer.Typer(
    name="bib-timing",
    help="Timing visual por dorsales (local, Linux/AMD).",
    no_args_is_help=True,
)
console = Console()


@app.command()
def smoke() -> None:
    """Verifica PyTorch / ROCm / CPU y un tensor en el dispositivo."""
    info = probe_device()
    table = Table(title="Dispositivo de inferencia")
    table.add_column("Campo")
    table.add_column("Valor")
    table.add_row("backend", info.backend)
    table.add_row("device", info.device)
    table.add_row("name", info.name)
    table.add_row("available", str(info.available))
    table.add_row("notes", info.notes)
    console.print(table)

    try:
        import torch

        if info.device.startswith("cuda"):
            x = torch.randn(1024, 1024, device=info.device)
            y = x @ x
            console.print(
                f"[green]OK[/green] matmul GPU: shape={tuple(y.shape)} "
                f"mean={float(y.mean()):.4f}"
            )
        else:
            x = torch.randn(256, 256)
            y = x @ x
            console.print(
                f"[yellow]OK CPU[/yellow] matmul: shape={tuple(y.shape)}"
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Falló smoke test:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("process")
def process_cmd(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Video de entrada"),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="Carpeta de salida"),
    line: Optional[str] = typer.Option(
        None,
        "--line",
        help="Línea de meta: x0,y0,x1,y1 en píxeles. Si se omite, línea horizontal al 55%.",
    ),
    model: str = typer.Option("yolo11n.pt", "--model", "-m"),
    conf: float = typer.Option(0.35, "--conf"),
    imgsz: int = typer.Option(640, "--imgsz"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames"),
    no_preview: bool = typer.Option(False, "--no-preview"),
    cpu: bool = typer.Option(False, "--cpu", help="Forzar CPU"),
) -> None:
    """Fase 1: detecta cruces de línea (personas) en video offline."""
    import cv2

    info = probe_device(prefer_gpu=not cpu)

    if line:
        parts = [float(p.strip()) for p in line.split(",")]
        if len(parts) != 4:
            console.print("[red]--line debe ser x0,y0,x1,y1[/red]")
            raise typer.Exit(code=2)
        finish = FinishLine(Point(parts[0], parts[1]), Point(parts[2], parts[3]))
    else:
        cap = cv2.VideoCapture(str(video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        finish = default_horizontal_line(w, h)
        console.print(
            f"[yellow]Sin --line:[/yellow] usando horizontal por defecto "
            f"y≈{int(h * 0.55)} px"
        )

    process_video(
        video,
        finish,
        output,
        model_name=model,
        conf=conf,
        imgsz=imgsz,
        max_frames=max_frames,
        device_info=info,
        save_preview=not no_preview,
    )


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8765, "--port"),
) -> None:
    """Interfaz web: subir video, calibrar línea de meta y procesar."""
    try:
        import uvicorn
    except ImportError as exc:
        console.print(
            "[red]Falta uvicorn/fastapi.[/red] Instalá: pip install 'bib-timing[web]'"
        )
        raise typer.Exit(code=1) from exc

    console.print(f"UI en [bold]http://{host}:{port}[/bold]")
    uvicorn.run(
        "bib_timing.web.app:app",
        host=host,
        port=port,
        reload=False,
    )


@app.command("convert")
def convert_cmd(
    video: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("data/clip_h264.mp4"), "--output", "-o"),
    start: float = typer.Option(0.0, "--start", help="Segundo de inicio"),
    seconds: Optional[float] = typer.Option(
        90.0, "--seconds", "-t", help="Duración. Usá 0 para el video completo."
    ),
) -> None:
    """Convierte a H.264 (útil si el origen es AV1/VP9 de YouTube)."""
    dur = None if seconds == 0 else seconds
    convert_to_h264(video, output, start_sec=start, seconds=dur)


@app.command("pick-line")
def pick_line(
    video: Path = typer.Argument(..., exists=True, readable=True),
    frame: int = typer.Option(0, "--frame", "-f", help="Frame a mostrar"),
) -> None:
    """Ayuda a calibrar la línea: imprime tamaño del frame y sugiere --line."""
    import cv2

    cap = open_video(video)
    if frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
    else:
        ok, img = cap.read()
    cap.release()
    if not ok:
        console.print("[red]No se pudo leer el frame[/red]")
        raise typer.Exit(code=1)

    h, w = img.shape[:2]
    y = int(h * 0.55)
    suggested = f"0,{y},{w},{y}"
    out = Path("output") / "calibration_frame.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.line(img, (0, y), (w, y), (0, 255, 255), 2)
    cv2.imwrite(str(out), img)
    console.print(f"Frame {w}x{h} → {out}")
    console.print(f"Sugerencia: --line {suggested}")
    console.print(
        "Ajustá los puntos mirando la imagen y pasá --line x0,y0,x1,y1 a process."
    )


if __name__ == "__main__":
    app()
