"""Persistencia simple de eventos de cruce (JSONL + CSV)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CrossingEvent:
    track_id: int
    frame: int
    time_sec: float
    x: float
    y: float
    confidence: float
    bib: str | None = None
    bib_confidence: float | None = None
    bib_votes: str | None = None  # ej. "132:8,182:2,732:1"
    ocr_samples: int = 0
    source_video: str = ""
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    snapshot: str | None = None
    crop: str | None = None  # mejor recorte de dorsal (relativo al job)


def save_events_jsonl(path: Path, events: list[CrossingEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")


def save_events_csv(path: Path, events: list[CrossingEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "track_id",
        "frame",
        "time_sec",
        "x",
        "y",
        "confidence",
        "bib",
        "bib_confidence",
        "bib_votes",
        "ocr_samples",
        "x1",
        "y1",
        "x2",
        "y2",
        "snapshot",
        "crop",
        "source_video",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ev in events:
            writer.writerow(asdict(ev))
