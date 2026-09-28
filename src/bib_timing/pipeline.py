"""Pipeline: tracking + cruce de meta + OCR multi-frame con votación."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.console import Console
from ultralytics import YOLO

from bib_timing.device import DeviceInfo, probe_device
from bib_timing.finish_line import DetectionZone, FinishLine, Point, crossed_line
from bib_timing.ocr import BibVotes
from bib_timing.results import CrossingEvent, save_events_csv, save_events_jsonl
from bib_timing.video_io import open_video

console = Console()

PERSON_CLASS = 0

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]


@dataclass
class TrackState:
    track_id: int
    last_centroid: Point | None = None
    last_box: tuple[float, float, float, float] | None = None
    last_score: float = 0.0
    last_seen_frame: int = -1
    crossed: bool = False
    cross_frame: int | None = None
    cross_time_sec: float | None = None
    cross_point: Point | None = None
    cross_box: tuple[float, float, float, float] | None = None
    cross_score: float = 0.0
    votes: BibVotes = field(default_factory=BibVotes)
    crop_count: int = 0
    finalized: bool = False
    # frame BGR del mejor OCR (para snapshot final)
    best_frame_img: np.ndarray | None = None
    best_box: tuple[float, float, float, float] | None = None
    best_crop_img: np.ndarray | None = None
    was_in_zone: bool = False
    in_zone: bool = False
    zone_exit_frame: int | None = None


def _annotate_detection(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    label: str,
    *,
    color: tuple[int, int, int] = (0, 220, 255),
) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, (x2 - x1) / 280)
    thickness = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    ty = max(0, y1 - 8)
    cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 8, ty + 4), color, -1)
    cv2.putText(
        out,
        label,
        (x1 + 4, ty),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return out


def _format_votes(counts: dict[str, int]) -> str:
    parts = [f"{b}:{c}" for b, c in sorted(counts.items(), key=lambda t: -t[1])]
    return ",".join(parts)


def process_video(
    video_path: Path,
    finish_line: FinishLine,
    output_dir: Path,
    *,
    model_name: str = "yolo11n.pt",
    bib_model: str | Path | None = None,
    conf: float = 0.35,
    imgsz: int = 640,
    max_frames: int | None = None,
    device_info: DeviceInfo | None = None,
    save_preview: bool = True,
    enable_ocr: bool = True,
    known_bibs: set[str] | None = None,
    detection_zone: DetectionZone | None = None,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    live_every_n: int = 3,
    lost_patience: int | None = None,
    save_crops: bool = True,
) -> list[CrossingEvent]:
    """
    - Dentro del área de detección: ROI dorsal + OCR → votos por track.
    - Cruce de línea: marca tiempo.
    - Al salir del área (o de cámara): cierra dorsal por mayoría.
    """
    device_info = device_info or probe_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir = output_dir / "snapshots"
    crops_dir = output_dir / "crops"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
    live_path = output_dir / "live.jpg"
    cancelled = False

    bib_weights: Path | None = None
    if bib_model:
        bib_weights = Path(bib_model)
        if not bib_weights.exists():
            console.print(f"[yellow]Modelo bib no encontrado:[/yellow] {bib_weights}")
            bib_weights = None

    console.print(f"[bold]Dispositivo:[/bold] {device_info.backend} · {device_info.name}")
    console.print(f"[bold]Video:[/bold] {video_path}")
    if bib_weights:
        console.print(f"[bold]Detector dorsal:[/bold] {bib_weights}")
    if known_bibs:
        console.print(f"[bold]Reconocimiento:[/bold] {len(known_bibs)} números conocidos")
    console.print(
        f"[bold]Línea de meta:[/bold] "
        f"({finish_line.a.x:.0f},{finish_line.a.y:.0f}) → "
        f"({finish_line.b.x:.0f},{finish_line.b.y:.0f})"
    )
    if detection_zone is not None:
        z = detection_zone.normalized()
        console.print(
            f"[bold]Área detección:[/bold] "
            f"({z.x0:.0f},{z.y0:.0f})–({z.x1:.0f},{z.y1:.0f})"
        )

    model = YOLO(model_name)
    cap = open_video(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and total > 0:
        total = min(total, max_frames)

    if lost_patience is None:
        lost_patience = max(20, int(fps * 0.7))

    ocr_gpu = device_info.device.startswith("cuda")
    if enable_ocr:
        try:
            from bib_timing.ocr import body_crop, read_bib_scored  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]OCR deshabilitado:[/yellow] {exc}")
            enable_ocr = False

    writer = None
    preview_path = output_dir / "preview_crossings.mp4"
    if save_preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))

    tracks: dict[int, TrackState] = {}
    events: list[CrossingEvent] = []
    frame_idx = 0
    live_seq = 0

    def emit(msg: str, *, extra: dict[str, Any] | None = None) -> None:
        if on_progress is None:
            return
        pending = [
            {
                "track_id": t.track_id,
                "crossed": t.crossed,
                "cross_time_sec": t.cross_time_sec,
                "ocr_samples": len(t.votes.hits),
                "provisional_bib": t.votes.winner()[0],
                "votes": dict(t.votes.counts),
            }
            for t in tracks.values()
            if t.crossed and not t.finalized
        ]
        payload: dict[str, Any] = {
            "frame": frame_idx,
            "total": total,
            "message": msg,
            "events": [e.__dict__ for e in events],
            "pending": pending,
            "live_seq": live_seq,
        }
        if extra:
            payload.update(extra)
        on_progress(payload)

    def finalize_track(st: TrackState, *, reason: str) -> None:
        if st.finalized:
            return
        st.finalized = True
        bib, bib_conf, counts = st.votes.winner()
        votes_str = _format_votes(counts) if counts else None

        if not st.crossed:
            console.print(
                f"[dim]Track {st.track_id} finalizado ({reason}) sin cruce · "
                f"ocr={len(st.votes.hits)} bib={bib or '—'}[/dim]"
            )
            return

        box = st.best_box or st.cross_box or st.last_box
        label = f"BIB {bib}" if bib else f"ID {st.track_id}"
        snap_name = (
            f"track{st.track_id}_t{st.cross_time_sec:.2f}"
            + (f"_bib{bib}" if bib else "")
            + ".jpg"
        )
        snap_rel = None
        crop_rel = None
        if box is not None:
            base = st.best_frame_img
            if base is not None:
                snap = _annotate_detection(base, box, label)
                snap_path = snaps_dir / snap_name
                cv2.imwrite(str(snap_path), snap, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                snap_rel = str(snap_path.relative_to(output_dir))
        if st.best_crop_img is not None and st.best_crop_img.size > 0:
            crop_name = (
                f"track{st.track_id}"
                + (f"_bib{bib}" if bib else "")
                + "_best.jpg"
            )
            crop_path = crops_dir / crop_name
            cv2.imwrite(
                str(crop_path),
                st.best_crop_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            crop_rel = str(crop_path.relative_to(output_dir))
        # Contexto persona (sin anotación) para refinar detector
        if st.best_frame_img is not None and st.best_box is not None:
            from bib_timing.ocr import body_crop

            person = body_crop(st.best_frame_img, st.best_box)
            if person.size > 0:
                ctx_name = f"track{st.track_id}_person.jpg"
                ctx_path = crops_dir / ctx_name
                cv2.imwrite(
                    str(ctx_path), person, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                # si no había crop de dorsal, usar persona
                if crop_rel is None:
                    crop_rel = str(ctx_path.relative_to(output_dir))

        assert st.cross_frame is not None and st.cross_time_sec is not None
        assert st.cross_point is not None
        x1 = y1 = x2 = y2 = None
        if st.cross_box:
            x1, y1, x2, y2 = st.cross_box

        ev = CrossingEvent(
            track_id=st.track_id,
            frame=st.cross_frame,
            time_sec=round(st.cross_time_sec, 3),
            x=st.cross_point.x,
            y=st.cross_point.y,
            confidence=float(st.cross_score),
            bib=bib,
            bib_confidence=round(bib_conf, 3) if bib else None,
            bib_votes=votes_str,
            ocr_samples=len(st.votes.hits),
            source_video=str(video_path),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            snapshot=snap_rel,
            crop=crop_rel,
        )
        events.append(ev)
        console.print(
            f"[green]Dorsal confirmado[/green] track={st.track_id} "
            f"bib={bib or '—'} votos={votes_str or '—'} "
            f"t_meta={st.cross_time_sec:.3f}s ({reason})"
        )
        emit(
            f"Dorsal track={st.track_id}: {bib or 'sin lectura'}",
            extra={"new_snapshot": snap_name if snap_rel else None},
        )

    try:
        while True:
            if should_cancel is not None and should_cancel():
                cancelled = True
                console.print("[yellow]Procesamiento detenido por el usuario[/yellow]")
                emit("Detenido por el usuario", extra={"cancelled": True})
                break

            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            results = model.track(
                frame,
                persist=True,
                conf=conf,
                imgsz=imgsz,
                classes=[PERSON_CLASS],
                tracker="bytetrack.yaml",
                device=device_info.device,
                verbose=False,
            )
            result = results[0]

            annotated = result.plot()
            cv2.line(
                annotated,
                (int(finish_line.a.x), int(finish_line.a.y)),
                (int(finish_line.b.x), int(finish_line.b.y)),
                (0, 255, 255),
                2,
            )
            zone = detection_zone
            if zone is None:
                zone = DetectionZone.full_frame(width, height)
            else:
                zone = zone.normalized()
            overlay = annotated.copy()
            cv2.rectangle(
                overlay,
                (int(zone.x0), int(zone.y0)),
                (int(zone.x1), int(zone.y1)),
                (255, 180, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
            cv2.rectangle(
                annotated,
                (int(zone.x0), int(zone.y0)),
                (int(zone.x1), int(zone.y1)),
                (255, 180, 0),
                2,
            )

            seen_now: set[int] = set()

            if result.boxes is not None and result.boxes.id is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for box, tid, score in zip(xyxy, ids, confs):
                    tid = int(tid)
                    seen_now.add(tid)
                    x1, y1, x2, y2 = (float(v) for v in box)
                    box_t = (x1, y1, x2, y2)
                    cx = (x1 + x2) / 2
                    cy = y1 + 0.65 * (y2 - y1)
                    curr = Point(cx, cy)
                    in_zone = zone.contains_box(box_t)

                    st = tracks.get(tid)
                    if st is None:
                        st = TrackState(track_id=tid)
                        tracks[tid] = st

                    st.in_zone = in_zone
                    if in_zone:
                        st.was_in_zone = True
                        st.zone_exit_frame = None
                    elif st.was_in_zone and st.zone_exit_frame is None:
                        st.zone_exit_frame = frame_idx

                    # --- cruce de línea (solo tiempo; dorsal después) ---
                    if (
                        st.last_centroid is not None
                        and not st.crossed
                        and crossed_line(st.last_centroid, curr, finish_line)
                    ):
                        st.crossed = True
                        st.cross_frame = frame_idx
                        st.cross_time_sec = frame_idx / fps
                        st.cross_point = curr
                        st.cross_box = box_t
                        st.cross_score = float(score)
                        console.print(
                            f"[cyan]Cruce[/cyan] track={tid} "
                            f"t={st.cross_time_sec:.3f}s — "
                            f"acumulando OCR en área de detección"
                        )
                        cv2.circle(annotated, (int(cx), int(cy)), 12, (0, 0, 255), -1)
                        cv2.putText(
                            annotated,
                            f"#{tid} META",
                            (int(cx) + 10, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        emit(f"Cruce track={tid} — OCR mientras esté en el área")

                    # --- ROI dorsal + OCR solo dentro del área ---
                    if in_zone and enable_ocr:
                        from bib_timing.ocr import (
                            body_crop,
                            read_bib_scored,
                            refine_bib_with_known,
                        )
                        from bib_timing.training.bib_detector import propose_bib_roi
                        from bib_timing.training.bib_roi import crop_proposal

                        person = body_crop(frame, box_t)
                        crop = person
                        if person.size > 0:
                            prop = propose_bib_roi(
                                person,
                                bib_weights=bib_weights,
                                conf=max(0.15, conf - 0.1),
                                device=device_info.device,
                            )
                            if prop is not None:
                                crop = crop_proposal(person, prop)
                                if crop.size == 0:
                                    crop = person
                        if save_crops and crop.size > 0:
                            tdir = crops_dir / f"track_{tid}"
                            tdir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(
                                str(tdir / f"frame_{frame_idx:06d}.jpg"),
                                crop,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                            )
                        bib, ocr_score = read_bib_scored(crop, use_gpu=ocr_gpu)
                        bib, ocr_score = refine_bib_with_known(
                            bib, ocr_score, known_bibs
                        )
                        prev_best = st.votes.best_hit
                        st.votes.add(bib, ocr_score, frame_idx)
                        st.crop_count += 1
                        if bib and (
                            prev_best is None
                            or (
                                st.votes.best_hit is not None
                                and st.votes.best_hit.frame == frame_idx
                            )
                        ):
                            st.best_frame_img = frame.copy()
                            st.best_box = box_t
                            if crop.size > 0:
                                st.best_crop_img = crop.copy()

                        provisional = st.votes.winner()[0]
                        if provisional:
                            cv2.putText(
                                annotated,
                                f"#{tid}?{provisional}",
                                (int(x1), max(20, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 180),
                                2,
                            )
                    elif in_zone and save_crops:
                        from bib_timing.ocr import body_crop

                        crop = body_crop(frame, box_t)
                        if crop.size > 0:
                            tdir = crops_dir / f"track_{tid}"
                            tdir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(
                                str(tdir / f"frame_{frame_idx:06d}.jpg"),
                                crop,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                            )
                            st.crop_count += 1
                            if st.best_crop_img is None:
                                st.best_crop_img = crop.copy()
                                st.best_frame_img = frame.copy()
                                st.best_box = box_t

                    if in_zone:
                        cv2.rectangle(
                            annotated,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 128),
                            2,
                        )

                    st.last_centroid = curr
                    st.last_box = box_t
                    st.last_score = float(score)
                    st.last_seen_frame = frame_idx

            # Salida del área de detección → finalizar
            for tid, st in list(tracks.items()):
                if st.finalized:
                    continue
                if st.was_in_zone and st.zone_exit_frame is not None:
                    if frame_idx - st.zone_exit_frame >= lost_patience:
                        finalize_track(st, reason="salió del área de detección")
                        continue
                # Ausente de cámara
                if tid not in seen_now and frame_idx - st.last_seen_frame >= lost_patience:
                    finalize_track(st, reason="salió de cámara")

            if writer is not None:
                writer.write(annotated)

            if frame_idx % max(1, live_every_n) == 0:
                cv2.imwrite(
                    str(live_path),
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                )
                live_seq += 1
                pct = (
                    f"{100.0 * frame_idx / total:.0f}%"
                    if total > 0
                    else f"frame {frame_idx}"
                )
                n_pending = sum(1 for t in tracks.values() if t.crossed and not t.finalized)
                emit(
                    f"Analizando {pct} · pendientes dorsal: {n_pending}"
                )

            frame_idx += 1
            if frame_idx % 100 == 0:
                console.print(f"Procesados {frame_idx} frames…")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # fin de video / cancel: cerrar tracks abiertos
    for st in tracks.values():
        if not st.finalized:
            finalize_track(
                st,
                reason="fin de video" if not cancelled else "detenido",
            )

    save_events_jsonl(output_dir / "crossings.jsonl", events)
    save_events_csv(output_dir / "crossings.csv", events)
    if cancelled:
        console.print(
            f"[yellow]Detenido:[/yellow] {len(events)} cruces → "
            f"{output_dir}/crossings.csv"
        )
    else:
        console.print(
            f"[bold]Listo:[/bold] {len(events)} cruces → {output_dir}/crossings.csv"
        )
    if save_preview:
        console.print(f"Preview: {preview_path}")
    if save_crops:
        console.print(f"Recortes: {crops_dir}")
    emit(
        "Detenido" if cancelled else "Completado",
        extra={"done": not cancelled, "cancelled": cancelled},
    )
    return events


def default_horizontal_line(width: int, height: int, y_ratio: float = 0.55) -> FinishLine:
    y = height * y_ratio
    return FinishLine(Point(width * 0.1, y), Point(width * 0.9, y))
