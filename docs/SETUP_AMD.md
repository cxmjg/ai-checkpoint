# Setup AMD ROCm — RX 6600 XT (CachyOS / Arch)

## Contexto

| Dato | Valor |
| --- | --- |
| GPU | AMD Radeon RX 6600 XT |
| Arquitectura | RDNA2 · **gfx1032** |
| Soporte ROCm | **No oficial** (gfx1030 sí lo es) |
| Workaround | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |

El proyecto usa PyTorch ROCm. Ultralytics habla con la GPU vía `device=0` /
`cuda:0` (misma API que CUDA; no existe `device=rocm`).

## Problema típico (lo que viste en el smoke test)

Si corrés `pip install -e .` **sin** instalar PyTorch ROCm antes, Ultralytics
baja `torch` desde PyPI → build **CUDA/NVIDIA** (`+cu130`). En AMD el smoke
reporta:

```
backend: cpu
notes: GPU no disponible
```

Verificación rápida:

```fish
python -c "import torch; print(torch.__version__); print(getattr(torch.version,'hip',None)); print(torch.version.cuda)"
```

- Mal: `2.14.0+cu130`, `hip=None`, `cuda=13.0`
- Bien: versión con `+rocm…`, `hip` no nulo, `cuda` suele ser `None` o un string de compat

## Shell: fish

```fish
source .venv/bin/activate.fish
set -x HSA_OVERRIDE_GFX_VERSION 10.3.0
```

(No uses `source .venv/bin/activate`: es script de bash.)

## Python: evitá 3.14 para ROCm

En CachyOS el default suele ser **Python 3.14**. Los wheels ROCm estables
suelen llegar primero a **3.12 / 3.13**. Recomendado:

```fish
sudo pacman -S python313
```

## Instalación limpia (recomendada)

```fish
cd ~/Documentos/Repositorios/AI_Checkpoint

# 1) Borrar el venv CUDA actual
deactivate  # si estaba activo
rm -rf .venv

# 2) Crear venv con 3.13
python3.13 -m venv .venv
source .venv/bin/activate.fish
set -x HSA_OVERRIDE_GFX_VERSION 10.3.0
pip install -U pip

# 3) PyTorch ROCm PRIMERO (ajustá el índice a tu ROCm del sistema)
# Ejemplos: rocm6.4 | rocm7.0 | rocm7.1 | nightly/rocm7.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

# 4) Resto del proyecto SIN reemplazar torch
pip install -e . --no-deps
pip install ultralytics opencv-python-headless numpy typer rich pydantic pydantic-settings

# 5) Verificar
python -c "import torch; print(torch.__version__, getattr(torch.version,'hip',None), torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
python -m bib_timing.cli smoke
```

Si el índice `rocm6.4` no tiene wheel para tu Python, probá `rocm7.0` /
`rocm7.1`, o usá Docker (abajo).

## Si ya instalaste el torch CUDA (recuperación in-place)

```fish
source .venv/bin/activate.fish
set -x HSA_OVERRIDE_GFX_VERSION 10.3.0

pip uninstall -y torch torchvision torchaudio triton
# limpia restos NVIDIA (opcional pero recomendable)
pip freeze | string match -r '^(nvidia-|cuda-)' | string split -f1 '==' | xargs -r pip uninstall -y

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
python -m bib_timing.cli smoke
```

Si `pip` dice *No matching distribution* en Python 3.14: recreá el venv con
`python3.13` (sección anterior).

## Drivers ROCm en el sistema

```fish
lspci | grep -i vga
# sudo pacman -S rocm-hip-sdk   # o el meta-paquete que uses en CachyOS
rocminfo | grep -E 'Marketing Name|Name:.*gfx'
```

Usuario en grupos `video` / `render`. Acceso a `/dev/kfd` y `/dev/dri`.

## Fallback CPU

El CLI funciona sin GPU (más lento):

```fish
python -m bib_timing.cli process video.mp4 --cpu --max-frames 200
```

## Opción Docker (más estable en AMD)

```fish
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -v "$PWD":/workspace -w /workspace \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

Dentro: `pip install -e . --no-deps` + deps livianas, luego `smoke`.

## Problemas frecuentes

| Síntoma | Qué hacer |
| --- | --- |
| `+cu130` / backend cpu | Reinstalar torch desde índice ROCm; no desde PyPI |
| `cuda.is_available() == False` con `+rocm` | Override gfx, grupos video/render, `rocminfo` |
| `HIP error: invalid device function` | Versión ROCm ≠ wheels; probar otro índice o Docker |
| `No matching distribution` (cp314) | Usar Python 3.13 |
| VRAM llena (8 GB) | `yolo11n`, `imgsz=640` |
