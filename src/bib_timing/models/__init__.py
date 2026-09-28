"""Registro y entrenamiento de modelos de dorsal."""

from bib_timing.models.registry import (
    ModelInfo,
    create_model,
    get_model,
    list_models,
    load_recognize_known_bibs,
    models_root,
    resolve_weights,
    update_model,
)

__all__ = [
    "ModelInfo",
    "create_model",
    "get_model",
    "list_models",
    "load_recognize_known_bibs",
    "models_root",
    "resolve_weights",
    "update_model",
]
