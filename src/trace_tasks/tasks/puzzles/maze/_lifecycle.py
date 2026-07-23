"""Neutral rendering and output plumbing for maze public tasks."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable
from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from .shared.defaults import (
    font_trace_record,
    maze_style_trace,
    resolve_render_params,
    sample_maze_font,
)
from .shared.output import build_trace_payload
from .shared.prompts import build_maze_prompt_artifacts
from .shared.rendering import render_maze_exit_scene
from .shared.state import DOMAIN, SCENE_ID, SCENE_VARIANTS, TARGET_REACHABILITY_DESCRIPTIONS

_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def retry_maze_generation(
    *,
    build_case: Callable[[int, Mapping[str, Any], int], TaskOutput],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Retry task-local maze construction when semantic constraints reject a seed."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            return build_case(
                instance_seed=int(instance_seed) + int(attempt_index),
                params=params,
                max_attempts=int(max_attempts),
            )
        except ValueError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("maze generation failed without a captured error")


def resolve_maze_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    task_id: str,
    namespace: str,
) -> tuple[str, dict[str, float], Mapping[str, Any]]:
    """Resolve the public no-branch query id and return sanitized params."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=(SINGLE_QUERY_ID,),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(task_id),
        namespace=f"{namespace}.branch",
    )
    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported maze public branch: {selected_branch}")
    return str(selected_branch), dict(branch_probabilities), task_params


def resolve_maze_scene_variant(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Sample one maze visual variant uniformly unless explicitly pinned."""

    explicit = params.get("scene_variant")
    if explicit is not None:
        selected = str(explicit).strip()
        if selected not in set(SCENE_VARIANTS):
            raise ValueError(f"unsupported scene_variant: {explicit}")
        return str(selected), support_probability_map(SCENE_VARIANTS, selected=selected, sort_keys=True)
    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        SCENE_VARIANTS,
        sort_keys=True,
    )
    return str(selected), dict(probabilities)


def object_description_for_scene(scene_variant: str) -> str:
    """Return concise scene wording for one maze visual variant."""

    descriptions = {
        "classic_wall_maze": "an orthogonal wall maze with a START cell and labeled exits on the outer boundary",
        "paper_labyrinth_maze": "a paper-style orthogonal wall maze with a START cell and labeled exits on the outer boundary",
        "block_wall_maze": "a heavy-wall orthogonal maze with a START cell and labeled exits on the outer boundary",
    }
    return descriptions.get(str(scene_variant), descriptions["classic_wall_maze"])


def prepare_maze_visual_case(
    *,
    dataset: Mapping[str, Any],
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
    prompt_task_key: str,
    prompt_query_key: str,
    prompt_dynamic_slots: Mapping[str, Any],
    namespace: str,
) -> Dict[str, Any]:
    """Render a sampled maze dataset and matching prompt artifacts."""

    render_params = resolve_render_params(
        params,
        render_defaults=rendering_defaults,
        instance_seed=int(instance_seed),
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    render_params = replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        floor_fill_rgb=tuple(int(value) for value in scene_style.option_fill_rgb),
        wall_color_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        border_color_rgb=tuple(int(value) for value in scene_style.panel_border_rgb),
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
        subtle_grid_rgb=tuple(int(value) for value in scene_style.notebook_line_rgb),
    )
    font_family = sample_maze_font(
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=rendering_defaults,
    )
    font_meta = font_trace_record(str(font_family))
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    with temporary_default_font_family(str(font_family)):
        rendered_scene = render_maze_exit_scene(
            background,
            dataset=dataset,
            scene_variant=str(scene_variant),
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    dynamic_slots = dict(prompt_dynamic_slots)
    dynamic_slots.setdefault("object_description", object_description_for_scene(str(scene_variant)))
    prompt_defaults, prompt_artifacts = build_maze_prompt_artifacts(
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    return {
        "image": image,
        "rendered_scene": rendered_scene,
        "render_params": render_params,
        "scene_variant": str(scene_variant),
        "background_meta": dict(background_meta),
        "scene_style_meta": dict(scene_style_meta),
        "post_noise_meta": dict(post_noise_meta),
        "font_family": str(font_family),
        "font_meta": dict(font_meta),
        "prompt_defaults": dict(prompt_defaults),
        "prompt_artifacts": prompt_artifacts,
    }


def render_spec_from_visual(visual: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the common render-spec trace section for a rendered maze."""

    render_params = visual["render_params"]
    rendered_scene = visual["rendered_scene"]
    return {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(visual["scene_variant"]),
        "background_style": dict(visual["background_meta"]),
        "scene_style": {
            **dict(visual["scene_style_meta"]),
            "maze": maze_style_trace(render_params),
            "font": dict(visual["font_meta"]),
        },
        "post_image_noise": dict(visual["post_noise_meta"]),
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "layout": "orthogonal_wall_maze_with_boundary_exit_labels",
        "wall_stroke_width_px": int(render_params.wall_stroke_width_px),
        "outer_wall_stroke_width_px": int(render_params.outer_wall_stroke_width_px),
        "exit_marker_radius_px": int(render_params.exit_marker_radius_px),
        "exit_marker_shape": str(render_params.exit_marker_shape),
        "text_style": {"font": dict(visual["font_meta"])},
        "unit_size_jitter": dict(render_params.unit_size_jitter),
    }


def maze_render_map(*, rendered_scene: Any, render_params: Any, annotation_source: str) -> Dict[str, Any]:
    """Build the common pixel render map for a rendered maze scene."""

    return with_puzzle_unit_size_jitter(
        {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "item_bboxes_px": {
                str(key): [round(float(value), 3) for value in bbox]
                for key, bbox in rendered_scene.item_bbox_map.items()
            },
            "item_points_px": {
                str(key): [round(float(value), 3) for value in point]
                for key, point in rendered_scene.item_point_map.items()
            },
            "cell_bboxes_px": {
                str(key): [round(float(value), 3) for value in bbox]
                for key, bbox in rendered_scene.cell_bbox_map.items()
            },
            "annotation_source": str(annotation_source),
        },
        render_params.unit_size_jitter,
    )


def build_maze_task_output(
    *,
    dataset: Mapping[str, Any],
    visual: Mapping[str, Any],
    public_query_id: str,
    branch_probabilities: Mapping[str, float],
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    prompt_query_key: str,
    semantic_params: Mapping[str, Any],
    relation_fields: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
) -> TaskOutput:
    """Assemble a TaskOutput after the public task binds answer and annotation."""

    rendered_scene = visual["rendered_scene"]
    prompt_artifacts = visual["prompt_artifacts"]
    scene_variant = str(visual["scene_variant"])
    supporting_item_ids = [str(value) for value in dataset["supporting_item_ids"]]
    target_reachability = dataset.get("target_reachability")
    annotation_type = str(annotation_gt.type)
    if annotation_type in {"point", "point_set"}:
        annotation_source = "item_points_px"
    elif annotation_type == "segment":
        annotation_source = "cell_bboxes_px"
    else:
        annotation_source = "item_bboxes_px"
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(public_query_id),
        params={
            "query_id_probabilities": dict(branch_probabilities),
            "prompt_query_key": str(prompt_query_key),
            "target_reachability": (
                str(target_reachability) if target_reachability is not None else None
            ),
            "scene_variant": scene_variant,
            "maze_rows": int(dataset["maze_rows"]),
            "maze_cols": int(dataset["maze_cols"]),
            "exit_count": int(dataset["exit_count"]),
            "reachable_exit_total": int(dataset["reachable_exit_total"]),
            **dict(semantic_params),
        },
    )
    render_map = maze_render_map(
        rendered_scene=rendered_scene,
        render_params=visual["render_params"],
        annotation_source=annotation_source,
    )
    trace_payload = build_trace_payload(
        scene_ir={
            "scene_kind": f"puzzle_maze_{scene_variant}",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "selected_branch": str(public_query_id),
                "semantic_rule": "move_through_open_corridors_from_start_walls_block_motion",
                "target_reachability": (
                    str(target_reachability) if target_reachability is not None else None
                ),
                "scene_variant": scene_variant,
                "answer_value": dataset["answer_value"],
                "reachable_exit_labels": [str(value) for value in dataset["reachable_exit_labels"]],
                "unreachable_exit_labels": [str(value) for value in dataset["unreachable_exit_labels"]],
                "supporting_item_ids": list(supporting_item_ids),
                **dict(relation_fields),
            },
        },
        semantic_spec=query_spec,
        render_spec=render_spec_from_visual(visual),
        render_map=render_map,
        execution_trace={
            "query_id": str(public_query_id),
            "internal_query_id": str(prompt_query_key),
            "target_reachability": (
                str(target_reachability) if target_reachability is not None else None
            ),
            "scene_variant": scene_variant,
            "question_format": str(prompt_query_key),
            "view_family": str(dataset["view_family"]),
            "topology_rule": str(dataset["topology_rule"]),
            "maze_rows": int(dataset["maze_rows"]),
            "maze_cols": int(dataset["maze_cols"]),
            "maze_rows_range": list(dataset["maze_rows_range"]),
            "maze_cols_range": list(dataset["maze_cols_range"]),
            "start_cell": list(dataset["start_cell"]),
            "open_edges": list(dataset["open_edges"]),
            "exits": [dict(exit_spec) for exit_spec in dataset["exits"]],
            "exit_count": int(dataset["exit_count"]),
            "exit_count_range": list(dataset["exit_count_range"]),
            "reachable_exit_total": int(dataset["reachable_exit_total"]),
            "reachable_exit_total_range": list(dataset["reachable_exit_total_range"]),
            "reachable_exit_labels": [str(value) for value in dataset["reachable_exit_labels"]],
            "unreachable_exit_labels": [str(value) for value in dataset["unreachable_exit_labels"]],
            "answer_value": dataset["answer_value"],
            "supporting_item_ids": list(supporting_item_ids),
            "supporting_annotation_source": str(annotation_source),
            "annotation_policy": str(dataset["annotation_policy"]),
            "query_id_probabilities": dict(branch_probabilities),
            "solver_trace": dict(dataset["solver_trace"]),
            **dict(execution_fields),
        },
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
        answer_gt=answer_gt.to_dict(),
        annotation_gt=annotation_gt.to_dict(),
        prompt_defaults=visual["prompt_defaults"],
        prompt_artifacts=prompt_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=visual["image"],
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "build_maze_task_output",
    "object_description_for_scene",
    "prepare_maze_visual_case",
    "render_spec_from_visual",
    "resolve_maze_public_branch",
    "resolve_maze_scene_variant",
    "retry_maze_generation",
]
