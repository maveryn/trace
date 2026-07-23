"""Annotation helpers for pages cycle diagrams."""

from __future__ import annotations

from typing import Dict, List

from .state import CycleCase, RenderedCycleScene


def answer_stage_bbox(case: CycleCase, rendered: RenderedCycleScene) -> List[float]:
    """Return the scalar bbox for the answer stage."""

    bbox = rendered.stage_bbox_map.get(str(case.answer_stage_bbox_id))
    if bbox is None:
        raise RuntimeError("cycle answer stage bbox missing from render map")
    return [round(float(value), 3) for value in bbox]


def stage_records(case: CycleCase, rendered: RenderedCycleScene) -> List[Dict[str, object]]:
    """Return stage specs with rendered boxes for trace metadata."""

    records: List[Dict[str, object]] = []
    for spec in case.stage_specs:
        stage_bbox_id = str(spec["stage_bbox_id"])
        label_bbox_id = str(spec["stage_label_bbox_id"])
        records.append(
            {
                **dict(spec),
                "bbox_px": list(rendered.stage_bbox_map[stage_bbox_id]),
                "label_bbox_px": list(rendered.stage_label_bbox_map[label_bbox_id]),
            }
        )
    return records
