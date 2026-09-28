# AI Checkpoint — Timing visual por dorsales

Sistema **local** (Linux + AMD) para detectar atletas en la línea de meta de
maratones/carreras, reconocer dorsales con IA y registrar el tiempo de llegada.

> Análisis interactivo y hoja de ruta: abrí el canvas
> `bib-timing-roadmap` en Cursor (junto al chat).

## Objetivo

Sustituir o complementar el conteo manual en meta con un pipeline de visión:

1. Video (archivo o cámara)
2. Detección + tracking de atletas
3. Evento de cruce de la línea de meta
4. Lectura del dorsal (OCR / detector de dígitos)
5. Registro de tiempo exportable (CSV / JSON)

## Stack elegido (MVP)

| Pieza | Tecnología | Por qué |
| --- | --- | --- |
| Lenguaje | Python 3.11+ | Ecosistema CV/IA maduro |
| Inferencia | PyTorch **ROCm** | GPU AMD (API `cuda:0`) |
| Detector | Ultralytics **YOLO11** | Rápido, tracking integrado (ByteTrack) |
| OCR (fase 2) | EasyOCR | Corre sobre PyTorch/ROCm; PaddleOCR más adelante |
| Video | OpenCV | Lectura/escritura simple |
| CLI | Typer + Rich | Entregas incrementales sin UI pesada |
| Resultados | CSV + JSONL | Fácil de auditar |

### GPU: Radeon RX 6600 XT

La tarjeta es **gfx1032** y **no está en la matriz oficial de ROCm**. En la
práctica suele funcionar con:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

Detalle completo: [docs/SETUP_AMD.md](docs/SETUP_AMD.md).

## Hoja de ruta (resumen)

| Fase | Entrega | Estado |
| --- | --- | --- |
| 0 | Scaffold + smoke test GPU/CPU | **Hecho** |
| 1 | Cruce de línea (YOLO persona + tracking), sin OCR | **Hecho (MVP)** |
| 2 | OCR del dorsal en crop multi-frame | Pendiente |
| 3 | Fine-tune detector de clase `bib` | Pendiente |
| 4 | Deduplicación, revisión humana, export “oficial” | Pendiente |
| 5 | Cámara real (cámara) + dashboard operador | Pendiente |
| 6 | Multi-cámara, sync NTP, producto | Pendiente |

Detalle: [docs/ROADMAP.md](docs/ROADMAP.md).

## Instalación rápida (fish + AMD)

**Importante:** no instales solo con `pip install -e .` — Ultralytics baja
PyTorch **CUDA** y en tu RX 6600 XT el smoke queda en CPU. Instalá torch ROCm
**antes**. Detalle: [docs/SETUP_AMD.md](docs/SETUP_AMD.md).

```fish
cd ~/Documentos/Repositorios/AI_Checkpoint

# Python 3.13 recomendado (3.14 suele no tener wheels ROCm)
# sudo pacman -S python313
rm -rf .venv
python3.13 -m venv .venv
source .venv/bin/activate.fish
set -x HSA_OVERRIDE_GFX_VERSION 10.3.0

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install -e . --no-deps
pip install ultralytics opencv-python-headless numpy typer rich pydantic pydantic-settings

python -m bib_timing.cli smoke
```

Si `smoke` muestra `backend: rocm` y el nombre de la GPU, estás listo.

## Interfaz web

```fish
pip install -e ".[web,ocr]"
set -x HSA_OVERRIDE_GFX_VERSION 10.3.0
python -m bib_timing.cli serve
# Abrí http://127.0.0.1:8765
```

En la UI podés subir un video (AV1 se convierte a H.264), elegir un clip de
`data/`, marcar la línea de meta con dos clics y lanzar el procesamiento.

## Modo entrenamiento

En la UI: pestaña **Entrenamiento**.

1. **Detección de dorsal** — escanea el video, propone recortes que parecen
   dorsal (color celeste/blanco) y vos marcás *Sí / No / Saltar*.
2. **Reconocimiento de dorsal** — propone el recorte y vos confirmás o corregís
   el número.

Los datos quedan en `data/training/`. **Exportar dataset** arma carpetas
`positive/` / `negative/` y labels YOLO en `bib_detect/yolo/` para entrenar
después un detector de clase `bib`.

## Uso del MVP (Fase 1) — CLI

```bash
# Calibrar línea de meta (genera frame con línea sugerida)
python -m bib_timing.cli pick-line /ruta/al/video.mp4

# Procesar video offline
python -m bib_timing.cli process /ruta/al/video.mp4 \
  --line 100,540,1820,540 \
  -o output/carrera1
```

Salida:

- `output/carrera1/crossings.csv` — eventos (tiempo, track_id, confianza)
- `output/carrera1/crossings.jsonl`
- `output/carrera1/preview_crossings.mp4` — video anotado

En esta fase el `track_id` es un ID de tracking, **no el dorsal**. El operador
puede anotar el dorsal a mano en el CSV. La Fase 2 automatiza la lectura.

## Estructura

```
src/bib_timing/
  cli.py          # smoke | pick-line | process
  device.py       # detección ROCm/CPU + override AMD
  finish_line.py  # geometría de cruce
  pipeline.py     # YOLO track + eventos
  results.py      # CSV/JSONL
docs/
  SETUP_AMD.md
  ROADMAP.md
```

## Limitaciones actuales (importante)

- No lee dorsales todavía (solo cruces de personas).
- El tiempo es relativo al **inicio del video** (no al pistoletazo), hasta
  sincronizar reloj de carrera.
- Atletas muy juntos pueden compartir tracks; siempre prever revisión humana.
- Precisión temporal ≈ 1 frame (a 30 fps ≈ 33 ms; a 60 fps ≈ 17 ms).

## Referencias

- [Ultralytics AMD / ROCm](https://docs.ultralytics.com/integrations/amd)
- [visual-race-timing](https://github.com/raceconditionrunning/visual-race-timing)
- [bib-tagger-python](https://github.com/chbornman/bib-tagger-python)
- [MarathonBibDetector](https://huggingface.co/Faraphel/MarathonBibDetector)
