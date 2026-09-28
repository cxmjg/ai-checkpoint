"""Utilidades de dispositivo GPU/CPU (ROCm vía API torch.cuda)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    backend: str  # "rocm" | "cuda" | "cpu"
    device: str  # "cuda:0" | "cpu"
    name: str
    available: bool
    notes: str = ""


def ensure_amd_override() -> None:
    """RX 6600 XT (gfx1032) suele necesitar emular gfx1030."""
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


def probe_device(prefer_gpu: bool = True) -> DeviceInfo:
    ensure_amd_override()
    try:
        import torch
    except ImportError:
        return DeviceInfo(
            backend="cpu",
            device="cpu",
            name="CPU (PyTorch no instalado)",
            available=False,
            notes="Instalá PyTorch ROCm o CPU según docs/SETUP_AMD.md",
        )

    if prefer_gpu and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        hip = getattr(torch.version, "hip", None)
        backend = "rocm" if hip else "cuda"
        return DeviceInfo(
            backend=backend,
            device="cuda:0",
            name=name,
            available=True,
            notes=f"HIP={hip}" if hip else "CUDA build",
        )

    return DeviceInfo(
        backend="cpu",
        device="cpu",
        name="CPU",
        available=True,
        notes="GPU no disponible; usando CPU (más lento)",
    )
