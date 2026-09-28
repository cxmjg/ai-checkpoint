"""Entrenamiento / refinamiento de detector YOLO clase bib."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bib_timing.models.registry import (
    get_model,
    model_dir,
    resolve_weights,
    save_model,
    update_model,
)
from bib_timing.training.dataset import export_detect_dataset, export_recognize_dataset

ProgressCb = Callable[[dict[str, Any]], None]


def train_detect_model(
    project_root: Path,
    model_id: str,
    *,
    epochs: int = 40,
    imgsz: int = 640,
    batch: int = 8,
    cpu: bool = False,
    on_progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Exporta dataset detect + fine-tune YOLO. Refina desde parent_id / pesos propios."""
    info = get_model(project_root, model_id)
    if info is None or info.kind != "detect":
        raise FileNotFoundError(f"Modelo detect no encontrado: {model_id}")

    # Resolver base ANTES de marcar training (resolve_weights con ready fallaría).
    base_weights = "yolo11n.pt"
    base_msg = "Entrenando desde yolo11n…"
    if info.parent_id:
        parent_w = resolve_weights(
            project_root, info.parent_id, require_ready=False
        )
        if parent_w is not None:
            base_weights = str(parent_w)
            base_msg = f"Refinando desde {info.parent_id}…"
        else:
            own = resolve_weights(project_root, model_id, require_ready=False)
            if own is not None:
                base_weights = str(own)
                base_msg = "Padre sin pesos; continuando desde pesos propios…"
            else:
                base_msg = "Padre sin pesos; partiendo de yolo11n…"
    else:
        existing = resolve_weights(project_root, model_id, require_ready=False)
        if existing is not None:
            base_weights = str(existing)
            base_msg = "Continuando entrenamiento desde pesos actuales…"

    info.status = "training"
    info.message = "Exportando dataset…"
    save_model(project_root, info)
    if on_progress:
        on_progress({"status": "training", "message": info.message, "model_id": model_id})

    export = export_detect_dataset(project_root)
    yolo_dir = Path(export["yolo_dir"])
    n_pos = int(export.get("positive", 0))
    if n_pos < 5:
        info.status = "error"
        info.message = f"Hacen falta al menos 5 positivos (hay {n_pos})"
        save_model(project_root, info)
        raise RuntimeError(info.message)

    # data.yaml con path absoluto (Ultralytics)
    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {yolo_dir.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: bib\n",
        encoding="utf-8",
    )

    info.message = base_msg
    save_model(project_root, info)
    if on_progress:
        on_progress({"status": "training", "message": info.message, "model_id": model_id})

    from bib_timing.device import probe_device
    from bib_timing.training.bib_detector import invalidate_bib_cache
    from ultralytics import YOLO

    device = probe_device(prefer_gpu=not cpu)
    device_arg = device.device if device.device.startswith("cuda") else "cpu"

    mdir = model_dir(project_root, "detect", model_id)
    runs_dir = mdir / "runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    model = YOLO(base_weights)
    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device_arg,
        project=str(mdir),
        name="runs",
        exist_ok=True,
        verbose=False,
        patience=max(10, epochs // 3),
    )
    elapsed = time.time() - t0

    best = mdir / "runs" / "weights" / "best.pt"
    last = mdir / "runs" / "weights" / "last.pt"
    src = best if best.exists() else last
    if not src.exists():
        info.status = "error"
        info.message = "Entrenamiento terminó sin weights"
        save_model(project_root, info)
        raise RuntimeError(info.message)

    dest = mdir / "weights.pt"
    # copy (no copy2) + touch: fuerza mtime nuevo y evita caché obsoleta
    shutil.copy(src, dest)
    dest.touch()
    invalidate_bib_cache(dest)

    metrics: dict[str, Any] = {
        "epochs": epochs,
        "positive": n_pos,
        "negative": int(export.get("negative", 0)),
        "yolo_negatives": int(export.get("yolo_negatives", 0)),
        "elapsed_sec": round(elapsed, 1),
        "base_weights": base_weights,
        "device": device_arg,
    }
    # Ultralytics metrics si están
    try:
        box = getattr(results, "results_dict", None) or {}
        if isinstance(box, dict):
            for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"):
                if key in box:
                    metrics[key.replace("metrics/", "").replace("(B)", "")] = round(
                        float(box[key]), 4
                    )
    except Exception:  # noqa: BLE001
        pass

    info.weights = "weights.pt"
    info.status = "ready"
    info.message = (
        f"Listo · {n_pos} pos · {int(export.get('yolo_negatives', 0))} neg fondo · "
        f"{epochs} epochs"
    )
    info.metrics = metrics
    update_model(project_root, info)
    if on_progress:
        on_progress(
            {
                "status": "ready",
                "message": info.message,
                "model_id": model_id,
                "metrics": metrics,
            }
        )
    return {"model": info.id, "weights": str(dest), "metrics": metrics}


def register_recognize_model(
    project_root: Path,
    model_id: str,
    *,
    detect_model_id: str | None = None,
    on_progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """
    Consolida etiquetas de reconocimiento en un snapshot versionado.
    No fine-tunea EasyOCR (pendiente); sirve como paquete de datos + vínculo
    al detector de dorsal usado para propuestas.
    """
    info = get_model(project_root, model_id)
    if info is None or info.kind != "recognize":
        raise FileNotFoundError(f"Modelo recognize no encontrado: {model_id}")

    info.status = "training"
    info.message = "Exportando etiquetas de reconocimiento…"
    save_model(project_root, info)
    if on_progress:
        on_progress({"status": "training", "message": info.message, "model_id": model_id})

    export = export_recognize_dataset(project_root)
    n = int(export.get("labeled", 0))
    if n < 1:
        info.status = "error"
        info.message = "No hay dorsales etiquetados aún"
        save_model(project_root, info)
        raise RuntimeError(info.message)

    mdir = model_dir(project_root, "recognize", model_id)
    snap = mdir / "dataset"
    if snap.exists():
        shutil.rmtree(snap)
    src = Path(export["recognize_dir"])
    shutil.copytree(src, snap)

    info.weights = None
    info.status = "ready"
    info.message = f"Snapshot · {n} dorsales etiquetados"
    info.metrics = {
        "labeled": n,
        "detect_model_id": detect_model_id,
        "dataset": "dataset",
    }
    if detect_model_id:
        info.notes = f"Detector asociado: {detect_model_id}"
    update_model(project_root, info)
    if on_progress:
        on_progress(
            {
                "status": "ready",
                "message": info.message,
                "model_id": model_id,
                "metrics": info.metrics,
            }
        )
    return {"model": info.id, "labeled": n, "dataset": str(snap)}
