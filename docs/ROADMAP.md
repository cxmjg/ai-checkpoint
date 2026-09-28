# Hoja de ruta — AI Checkpoint (timing por dorsales)

## Principio

Entregas cortas, cada una usable en una carrera de prueba. La precisión de
“resultado oficial” llega tarde; primero el pipeline y la calibración óptica.

## Fase 0 — Entorno (hecho)

- [x] Estructura Python (`src/bib_timing`)
- [x] Detección de dispositivo ROCm/CPU + override gfx1032
- [x] CLI `smoke`

**Criterio de salida:** `python -m bib_timing.cli smoke` OK en GPU o CPU.

## Fase 1 — Cruce de línea sin OCR (hecho / MVP)

- [x] YOLO persona + ByteTrack
- [x] Segmento de meta configurable (`--line`)
- [x] Export CSV/JSONL + preview MP4
- [x] `pick-line` para calibración

**Criterio de salida:** en un video de prueba, cada atleta que cruza genera un
evento con timestamp razonable; falsos positivos < 10% con línea bien colocada.

**Cómo probar:**

1. Grabar 10–30 s de gente cruzando una marca en el piso.
2. `pick-line` → ajustar `--line`.
3. `process` → revisar `crossings.csv` vs preview.

## Fase 2 — OCR de dorsal

- [ ] Crop de torso / bbox en frames alrededor del cruce (±N frames)
- [ ] EasyOCR (digits-only) + filtro regex `^\d{1,5}$`
- [ ] Votación: modo de lecturas entre frames
- [ ] Campo `bib` en CSV; cola `needs_review` si confianza baja

**Criterio:** ≥70% bibs correctos en set controlado (luz buena, dorsales
frontales). El resto a revisión manual.

## Fase 3 — Detector de dorsales

- [ ] Dataset (fotos propias + RBNR / similares)
- [ ] Fine-tune YOLO clase `bib`
- [ ] OCR solo sobre caja del dorsal
- [ ] Comparar vs Fase 2 (precision/recall)

**Criterio:** menos falsos positivos que OCR sobre persona completa.

## Fase 4 — Robustez operativa

- [ ] Histéresis / zona de meta (evitar doble conteo)
- [ ] Cooldown por track_id / bib
- [ ] UI mínima de verificación (Aceptar / Corregir / Descartar)
- [ ] Sync con hora de largada (offset configurable)
- [ ] Export formato ranking (posición, bib, tiempo neto)

**Criterio:** un operador puede cerrar resultados de una carrera chica (<200)
en <30 min incluyendo revisión.

## Fase 5 — Tiempo real

- [ ] Fuente USB/RTSP
- [ ] Pipeline streaming con cola
- [ ] Dashboard live (últimos cruces)
- [ ] Alerta si FPS cae / GPU falla → fallback

**Criterio:** latencia evento→pantalla < 2 s a 1080p30 con `yolo11n` en 6600 XT.

## Fase 6 — Producto

- [ ] Multi-cámara + sync NTP
- [ ] Calibración óptica guiada
- [ ] Integración opcional con chips RFID como ground truth
- [ ] Empaquetado (systemd service / contenedor)

## Métricas a medir desde Fase 1

| Métrica | Definición |
| --- | --- |
| Recall de cruces | atletas reales detectados / total que cruzaron |
| Precision de cruces | eventos válidos / eventos emitidos |
| Error temporal | \|t_sistema − t_manual\| |
| (Fase 2+) Accuracy bib | bibs correctos / cruces con bib legible |
| FPS efectivos | frames procesados / segundo |

## Decisiones abiertas (para más adelante)

1. ¿Resultado “asistido” (humano en el loop) o automático puro?
2. ¿Una cámara frontal basta o se suma lateral?
3. ¿El tiempo oficial lo da este sistema o solo el orden de llegada?

Recomendación: empezar como **asistente de meta** (Fases 1–4), no como
reemplazo certificado de chip timing.
