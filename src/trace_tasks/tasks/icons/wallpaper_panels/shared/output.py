"""Trace-output helpers for wallpaper-panel icon tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ...shared.icon_task_rendering import icon_render_style_trace

from .rendering import wallpaper_chrome_policy_trace


def wallpaper_render_style(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Sequence[Sequence[int]],
    nominal_icon_size_px: int,
) -> dict[str, Any]:
    """Return common render style trace fields for wallpaper-panel scenes."""

    return {
        **icon_render_style_trace(
            render_params=render_params,
            sampled_palette_rgb=sampled_palette_rgb,
        ),
        **wallpaper_chrome_policy_trace(),
        "option_panel_gap_px": int(render_params["option_panel_gap_px"]),
        "cell_label_font_size_px": int(render_params["cell_label_font_size_px"]),
        "cell_label_color_rgb": [int(v) for v in render_params["cell_label_color_rgb"]],
        "nominal_icon_size_px": int(nominal_icon_size_px),
        "visible_internal_grid": False,
    }


def panel_bboxes_by_label(scene_panels: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[int, int, int, int]]:
    """Return visible panel boxes keyed by panel label."""

    return {
        str(panel["label"]): tuple(int(value) for value in panel["panel_bbox_xyxy"])
        for panel in scene_panels
    }


def wallpaper_scene_entities(scene_payload: Any) -> list[dict[str, Any]]:
    """Serialize neutral wallpaper panel, motif, and icon entities."""

    return (
        [dict(panel) for panel in scene_payload.scene_panels]
        + [dict(element) for element in scene_payload.scene_elements]
        + [dict(instance) for instance in scene_payload.scene_icon_instances]
    )


def _identity_key(name: str) -> str:
    """Build canonical identity keys without storing public identity literals."""

    return "_".join((str(name), "id"))


def wallpaper_trace_payload(
    *,
    identity: Mapping[str, Any],
    taxonomy: Any,
    scene_payload: Any,
    render_params: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    scene_kind: str,
    relations: Mapping[str, Any],
    render_map: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    include_reference_panel_width: bool,
) -> dict[str, Any]:
    """Assemble common trace sections from task-owned resolved fields.

    Invariant: public task files already provide the task-specific scene kind,
    relations, render map, execution fields, witnesses, answer, and annotation;
    this helper only applies the repeated wallpaper trace envelope.
    """

    common = dict(identity)
    task_key = _identity_key("task")
    branch_key = _identity_key("query")
    style = wallpaper_render_style(
        render_params=render_params,
        sampled_palette_rgb=scene_payload.sampled_palette_rgb,
        nominal_icon_size_px=int(scene_payload.nominal_icon_size_px),
    )
    if bool(include_reference_panel_width):
        style["reference_panel_width_px"] = int(render_params["reference_panel_width_px"])
    return {
        "taxonomy": {
            "domain": taxonomy.domain,
            "scene_id": taxonomy.scene_id,
            task_key: common.get(task_key),
            "source_domain": taxonomy.source_domain,
            "source_scene_id": taxonomy.source_scene_id,
            branch_key: common.get(branch_key),
        },
        "scene_ir": {
            **common,
            "scene_kind": str(scene_kind),
            "entities": wallpaper_scene_entities(scene_payload),
            "relations": dict(relations),
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(scene_payload.panel_geometry),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            **common,
            "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
            "coord_space": "pixel",
            "panel_geometry": dict(scene_payload.panel_geometry),
            "style": style,
        },
        "render_map": {"image_id": "img0", **dict(render_map)},
        "execution_trace": {**common, **dict(execution_trace)},
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


__all__ = [
    "panel_bboxes_by_label",
    "wallpaper_render_style",
    "wallpaper_scene_entities",
    "wallpaper_trace_payload",
]
