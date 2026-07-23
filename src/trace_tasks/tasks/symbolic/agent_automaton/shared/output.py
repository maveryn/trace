"""Trace-shell helpers for symbolic agent-automaton rendering artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from ....shared.prompt_variants import build_prompt_query_spec
from ...shared.unit_size_jitter import with_symbolic_unit_size_jitter

from .state import AgentRenderBundle
from .rules import state_count_for_rule


def agent_run_execution_fields(*, rows: int, cols: int, steps: int, rule_variant: str, initial_grid: Any, final_grid: Any) -> dict[str, Any]:
    """Return common execution fields for an already simulated run."""

    return {
        "state_count": state_count_for_rule(str(rule_variant)),
        "steps": int(steps),
        "grid_rows": int(rows),
        "grid_cols": int(cols),
        "initial_grid": [list(row) for row in initial_grid],
        "final_grid": [list(row) for row in final_grid],
    }


def build_agent_trace_payload(
    *,
    scene_name: str,
    prompt_artifacts: Any,
    branch_name: str,
    params_payload: Mapping[str, Any],
    render_bundle: AgentRenderBundle,
    execution_record: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    render_map_extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the shared trace shell after task-owned bindings exist."""

    render_params = render_bundle.render_params
    rendered = render_bundle.rendered
    render_map = {
        "image_id": "img0",
        "scene_bbox_px": [int(value) for value in rendered.scene_bbox_px],
        "item_bboxes_px": {str(key): list(value) for key, value in rendered.item_bboxes.items()},
        "annotation_source": "item_bboxes_px",
        "layout_jitter": dict(rendered.layout_jitter),
        "scene_style": dict(rendered.style_metadata),
        **dict(render_map_extra),
    }
    return {
        "scene_ir": {
            "scene_kind": str(scene_name),
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(params_payload),
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(branch_name),
            params=dict(params_payload),
        ),
        "render_spec": {
            "scene_id": str(scene_name),
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(params_payload.get("scene_variant", "")),
            "background_style": dict(render_bundle.background_meta),
            "post_image_noise": dict(render_bundle.post_noise_meta),
            "scene_bbox_px": [int(value) for value in rendered.scene_bbox_px],
            "render_params": {
                "cell_size_px": int(render_params.cell_size_px),
                "grid_gap_px": int(render_params.grid_gap_px),
                "option_card_width_px": int(render_params.option_card_width_px),
                "option_card_height_px": int(render_params.option_card_height_px),
                "option_grid_cell_px": int(render_params.option_grid_cell_px),
            },
            "unit_size_jitter": dict(render_params.unit_size_jitter),
            "layout_jitter": dict(rendered.layout_jitter),
            "scene_style": dict(rendered.style_metadata),
        },
        "render_map": with_symbolic_unit_size_jitter(render_map, render_params.unit_size_jitter),
        "execution_trace": dict(execution_record),
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }
