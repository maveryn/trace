"""Scene-neutral trace fragments for named-grid icon tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.prompt_variants import PromptTraceArtifacts
from ...shared.icon_task_rendering import icon_render_style_trace

from .defaults import SCENE_ID
from .state import NamedGridScenePayload, RenderedGridIcon
from .styles import named_grid_style_trace


def scene_ir_fragment(
    scene: NamedGridScenePayload,
    *,
    scene_kind: str,
    entities: list[dict[str, Any]],
    relations: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the scene-level IR wrapper around task-owned relations."""

    return {
        "scene_kind": str(scene_kind),
        "scene_id": SCENE_ID,
        "entities": list(entities),
        "relations": dict(relations),
        "frames": {
            "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            "panels": dict(scene.panel_geometry),
        },
    }


def render_spec_fragment(scene: NamedGridScenePayload, *, render_params: Mapping[str, Any]) -> Dict[str, Any]:
    """Build named-grid render metadata shared by all objectives."""

    return {
        "canvas_size": list(scene.panel_geometry["canvas_size"]),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "panel_geometry": dict(scene.panel_geometry),
        "grid_bbox_xyxy": [int(value) for value in scene.grid_bbox_xyxy],
        "cell_size_px": int(scene.cell_size_px),
        "style": {
            **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=scene.sampled_palette_rgb),
            **named_grid_style_trace(render_params),
        },
    }


def cell_bbox_map(scene: NamedGridScenePayload, *, rows: int, cols: int) -> Dict[str, list[int]]:
    """Return one keyed cell bbox for every row/column address."""

    return {
        f"r{int(row) + 1}c{int(col) + 1}": [int(value) for value in scene.cell_bboxes_xyxy[int(row)][int(col)]]
        for row in range(int(rows))
        for col in range(int(cols))
    }


def object_bbox_map(scene: NamedGridScenePayload) -> Dict[str, list[int]]:
    """Return object bboxes keyed by stable rendered icon instance id."""

    return {
        str(icon.instance_id): [int(value) for value in icon.bbox_xyxy]
        for icon in scene.icons
    }


def render_map_fragment(
    scene: NamedGridScenePayload,
    *,
    rows: int,
    cols: int,
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the shared render map plus task-owned annotation aliases."""

    payload: Dict[str, Any] = {
        "image_id": "img0",
        "object_bboxes_px": object_bbox_map(scene),
        "cell_bboxes_px": cell_bbox_map(scene, rows=int(rows), cols=int(cols)),
    }
    if extra_fields:
        payload.update(dict(extra_fields))
    return payload


def cells_to_trace(cells: Sequence[Tuple[int, int]]) -> list[list[int]]:
    """Serialize zero-based grid cell coordinates for trace payloads."""

    return [[int(row), int(col)] for row, col in cells]


def counts_to_trace(counts: Sequence[int]) -> list[int]:
    """Serialize a line-count vector for trace payloads."""

    return [int(value) for value in counts]


def shape_counts_from_icons(icons: Sequence[RenderedGridIcon]) -> Dict[str, int]:
    """Count rendered icons by shape id."""

    counts = Counter(str(icon.shape_id) for icon in icons)
    return {str(key): int(value) for key, value in counts.items()}


def instance_ids(icons: Sequence[RenderedGridIcon]) -> Tuple[str, ...]:
    """Return stable rendered icon instance ids."""

    return tuple(str(icon.instance_id) for icon in icons)


def task_output_prompt_variants(prompt_artifacts: PromptTraceArtifacts) -> Dict[str, str]:
    """Normalize prompt variants for final TaskOutput construction."""

    return {str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()}


def build_named_grid_trace_payload(
    *,
    scene: NamedGridScenePayload,
    scene_kind: str,
    entities: Sequence[Mapping[str, Any]],
    relations: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    render_params: Mapping[str, Any],
    rows: int,
    cols: int,
    render_map_extra: Mapping[str, Any] | None,
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    annotation_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the common named-grid trace payload around task-owned fields."""

    return {
        "scene_ir": scene_ir_fragment(
            scene,
            scene_kind=str(scene_kind),
            entities=[dict(entity) for entity in entities],
            relations=dict(relations),
        ),
        "query_spec": dict(query_spec),
        "render_spec": render_spec_fragment(scene, render_params=render_params),
        "render_map": render_map_fragment(
            scene,
            rows=int(rows),
            cols=int(cols),
            extra_fields=render_map_extra,
        ),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(annotation_payload["projected_annotation"]),
    }


__all__ = [
    "build_named_grid_trace_payload",
    "cell_bbox_map",
    "cells_to_trace",
    "counts_to_trace",
    "instance_ids",
    "object_bbox_map",
    "render_map_fragment",
    "render_spec_fragment",
    "scene_ir_fragment",
    "shape_counts_from_icons",
    "task_output_prompt_variants",
]
