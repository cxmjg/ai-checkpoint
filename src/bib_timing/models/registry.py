"""Registro local de modelos entrenados (detección / reconocimiento)."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    id: str
    name: str
    kind: str  # detect | recognize
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parent_id: str | None = None
    status: str = "empty"  # empty | training | ready | error
    message: str = ""
    weights: str | None = None  # relativo al directorio del modelo
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def models_root(project_root: Path) -> Path:
    root = project_root / "data" / "models"
    root.mkdir(parents=True, exist_ok=True)
    (root / "detect").mkdir(exist_ok=True)
    (root / "recognize").mkdir(exist_ok=True)
    return root


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower()).strip("-")
    return (s or "modelo")[:40]


def model_dir(project_root: Path, kind: str, model_id: str) -> Path:
    return models_root(project_root) / kind / model_id


def _meta_path(project_root: Path, kind: str, model_id: str) -> Path:
    return model_dir(project_root, kind, model_id) / "meta.json"


def save_model(project_root: Path, info: ModelInfo) -> None:
    d = model_dir(project_root, info.kind, info.id)
    d.mkdir(parents=True, exist_ok=True)
    info.updated_at = time.time()
    (_meta_path(project_root, info.kind, info.id)).write_text(
        json.dumps(asdict(info), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_model(project_root: Path, kind: str, model_id: str) -> ModelInfo:
    path = _meta_path(project_root, kind, model_id)
    if not path.exists():
        raise FileNotFoundError(model_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelInfo(**data)


def get_model(project_root: Path, model_id: str) -> ModelInfo | None:
    root = models_root(project_root)
    for kind in ("detect", "recognize"):
        path = root / kind / model_id / "meta.json"
        if path.exists():
            return load_model(project_root, kind, model_id)
    return None


def list_models(
    project_root: Path, *, kind: str | None = None
) -> list[ModelInfo]:
    root = models_root(project_root)
    kinds = [kind] if kind else ["detect", "recognize"]
    items: list[ModelInfo] = []
    for k in kinds:
        base = root / k
        if not base.exists():
            continue
        for p in sorted(base.glob("*/meta.json"), reverse=True):
            data = json.loads(p.read_text(encoding="utf-8"))
            info = ModelInfo(**data)
            if k == "detect":
                info = heal_ready_if_weights(project_root, info)
            items.append(info)
    items.sort(key=lambda m: m.updated_at, reverse=True)
    return items


def create_model(
    project_root: Path,
    *,
    name: str,
    kind: str,
    parent_id: str | None = None,
    notes: str = "",
) -> ModelInfo:
    if kind not in {"detect", "recognize"}:
        raise ValueError("kind debe ser detect o recognize")
    if parent_id:
        parent = get_model(project_root, parent_id)
        if parent is None:
            raise FileNotFoundError(f"Modelo padre no encontrado: {parent_id}")
        if parent.kind != kind:
            raise ValueError("El modelo padre debe ser del mismo tipo")

    mid = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
    info = ModelInfo(
        id=mid,
        name=name.strip() or mid,
        kind=kind,
        parent_id=parent_id,
        status="empty",
        message="Sin pesos aún — etiquetá datos y entrená",
        notes=notes,
    )
    save_model(project_root, info)
    return info


def update_model(project_root: Path, info: ModelInfo) -> ModelInfo:
    save_model(project_root, info)
    return info


def find_weights_file(project_root: Path, info: ModelInfo) -> Path | None:
    """Busca el archivo de pesos en el directorio del modelo."""
    d = model_dir(project_root, info.kind, info.id)
    candidates: list[Path] = []
    if info.weights:
        candidates.append(d / info.weights)
    candidates.append(d / "weights.pt")
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def load_recognize_known_bibs(
    project_root: Path, model_id: str | None
) -> set[str]:
    """
    Números etiquetados del snapshot de reconocimiento.
    Sirve en operación para reforzar/validar lecturas OCR.
    """
    if not model_id:
        return set()
    info = get_model(project_root, model_id)
    if info is None or info.kind != "recognize" or info.status != "ready":
        return set()
    bibs: set[str] = set()
    snap = model_dir(project_root, "recognize", info.id) / "dataset"
    if snap.exists():
        for p in snap.glob("*.jpg"):
            # archivos: {bib}__{cid}.jpg
            head = p.name.split("__", 1)[0].strip()
            if head.isdigit():
                bibs.add(head)
    return bibs


def heal_ready_if_weights(project_root: Path, info: ModelInfo) -> ModelInfo:
    """
    Recupera modelos con pesos en disco pero status error/empty (o training
    colgado sin corrida activa en runs/).
    """
    if info.kind != "detect" or info.status == "ready":
        return info
    if info.status not in {"training", "error", "empty"}:
        return info
    path = find_weights_file(project_root, info)
    if path is None:
        return info
    # Entrenamiento en curso: Ultralytics escribe en runs/ — no tocar
    runs = model_dir(project_root, info.kind, info.id) / "runs"
    if info.status == "training" and runs.exists():
        # Si hay results.csv reciente (< 2 min), asumir entrenamiento vivo
        results = runs / "results.csv"
        if results.exists():
            age = time.time() - results.stat().st_mtime
            if age < 120:
                return info
        # runs de un train ya terminado: se puede sanear
    info.status = "ready"
    info.weights = path.name
    if not info.message or "Entrenando" in info.message or "Exportando" in info.message:
        info.message = "Listo"
    save_model(project_root, info)
    return info


def resolve_weights(
    project_root: Path,
    model_id: str | None,
    *,
    require_ready: bool = True,
) -> Path | None:
    """
    Ruta a weights.pt.

    - require_ready=True (escaneo/ops): solo ready (sana estados colgados).
    - require_ready=False (fine-tune): pesos en disco aunque status sea training.
    """
    if not model_id:
        return None
    info = get_model(project_root, model_id)
    if info is None:
        return None
    if require_ready:
        info = heal_ready_if_weights(project_root, info)
        if info.status != "ready":
            return None
    return find_weights_file(project_root, info)
