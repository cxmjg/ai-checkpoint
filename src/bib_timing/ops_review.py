"""Aplicar revisión post-operación a datasets de entrenamiento."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from bib_timing.training.dataset import (
    Candidate,
    TrainSession,
    new_session_id,
    save_session,
    session_dir,
)


def apply_job_reviews(
    project_root: Path,
    job_dir: Path,
    events: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    *,
    video_path: str = "",
) -> dict[str, Any]:
    """
    Crea sesiones de entrenamiento a partir de la revisión de cruces.

    reviews: [{track_id, verdict: correct|incorrect|skip, bib?: str}]
    - correct → recognize labeled + detect positive (si hay crop)
    - incorrect → detect negative
    """
    by_track = {int(e["track_id"]): e for e in events if "track_id" in e}
    detect_cands: list[Candidate] = []
    recognize_cands: list[Candidate] = []

    detect_sid = new_session_id()
    recognize_sid = new_session_id()
    detect_dir = session_dir(project_root, detect_sid)
    recognize_dir = session_dir(project_root, recognize_sid)

    n_ok = n_bad = n_skip = 0
    for rev in reviews:
        tid = int(rev["track_id"])
        verdict = str(rev.get("verdict") or "skip").lower()
        bib = (rev.get("bib") or "").strip() or None
        ev = by_track.get(tid)
        if ev is None:
            continue
        if verdict == "skip":
            n_skip += 1
            continue

        crop_rel = ev.get("crop")
        snap_rel = ev.get("snapshot")
        src_crop = (job_dir / crop_rel) if crop_rel else None
        src_snap = (job_dir / snap_rel) if snap_rel else None
        if src_crop is not None and not src_crop.exists():
            src_crop = None
        if src_snap is not None and not src_snap.exists():
            src_snap = None
        if src_crop is None and src_snap is None:
            continue

        cid = f"ops_{tid}_{uuid.uuid4().hex[:8]}"
        if verdict == "correct":
            n_ok += 1
            final_bib = bib or (ev.get("bib") or "").strip() or None
            # detect positive
            img_name = f"{cid}.jpg"
            src = src_crop or src_snap
            assert src is not None
            dest = detect_dir / "images" / img_name
            shutil.copy2(src, dest)
            ctx_rel = None
            if src_snap is not None:
                ctx_name = f"{cid}_ctx.jpg"
                shutil.copy2(src_snap, detect_dir / "context" / ctx_name)
                ctx_rel = f"context/{ctx_name}"
            detect_cands.append(
                Candidate(
                    id=cid,
                    kind="detect",
                    frame=int(ev.get("frame") or 0),
                    track_id=tid,
                    image_rel=f"images/{img_name}",
                    context_rel=ctx_rel,
                    box_xyxy=None,
                    person_w=0,
                    person_h=0,
                    score=float(ev.get("bib_confidence") or ev.get("confidence") or 0.5),
                    method="ops_review",
                    status="positive",
                    label_bib=final_bib,
                    reviewed_at=time.time(),
                )
            )
            if final_bib and src_crop is not None:
                rcid = f"{cid}_rec"
                rimg = f"{rcid}.jpg"
                shutil.copy2(src_crop, recognize_dir / "images" / rimg)
                recognize_cands.append(
                    Candidate(
                        id=rcid,
                        kind="recognize",
                        frame=int(ev.get("frame") or 0),
                        track_id=tid,
                        image_rel=f"images/{rimg}",
                        context_rel=None,
                        box_xyxy=None,
                        person_w=0,
                        person_h=0,
                        score=float(ev.get("bib_confidence") or 0.5),
                        method="ops_review",
                        status="labeled",
                        label_bib=final_bib,
                        reviewed_at=time.time(),
                    )
                )
        elif verdict == "incorrect":
            n_bad += 1
            img_name = f"{cid}.jpg"
            src = src_crop or src_snap
            assert src is not None
            shutil.copy2(src, detect_dir / "images" / img_name)
            ctx_rel = None
            if src_snap is not None:
                ctx_name = f"{cid}_ctx.jpg"
                shutil.copy2(src_snap, detect_dir / "context" / ctx_name)
                ctx_rel = f"context/{ctx_name}"
            detect_cands.append(
                Candidate(
                    id=cid,
                    kind="detect",
                    frame=int(ev.get("frame") or 0),
                    track_id=tid,
                    image_rel=f"images/{img_name}",
                    context_rel=ctx_rel,
                    box_xyxy=None,
                    person_w=0,
                    person_h=0,
                    score=float(ev.get("confidence") or 0.3),
                    method="ops_review",
                    status="negative",
                    reviewed_at=time.time(),
                )
            )

    detect_session = None
    recognize_session = None
    if detect_cands:
        detect_session = TrainSession(
            id=detect_sid,
            kind="detect",
            video_path=video_path,
            status="done",
            message=f"Revisión ops · {len(detect_cands)} etiquetas",
            candidates=detect_cands,
        )
        save_session(project_root, detect_session)
    if recognize_cands:
        recognize_session = TrainSession(
            id=recognize_sid,
            kind="recognize",
            video_path=video_path,
            status="done",
            message=f"Revisión ops · {len(recognize_cands)} dorsales",
            candidates=recognize_cands,
        )
        save_session(project_root, recognize_session)

    return {
        "correct": n_ok,
        "incorrect": n_bad,
        "skipped": n_skip,
        "detect_session": detect_session.id if detect_session else None,
        "recognize_session": recognize_session.id if recognize_session else None,
        "detect_labels": len(detect_cands),
        "recognize_labels": len(recognize_cands),
    }
