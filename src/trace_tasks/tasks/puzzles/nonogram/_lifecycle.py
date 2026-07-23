"""Neutral rendering and output plumbing for nonogram public tasks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family

from .shared.defaults import (
    font_trace_record,
    resolve_render_params,
    resolve_scene_variant,
    sample_nonogram_font,
)
from .shared.output import build_trace_payload
from .shared.prompts import build_nonogram_prompt_artifacts
from .shared.rendering import render_nonogram_scene
from .shared.state import DOMAIN, NonogramDataset, SCENE_ID
from .shared.annotations import option_panel_bbox

_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15)


def retry_nonogram_generation(
    *,
    build_case: Callable[[int, Mapping[str, Any], int], TaskOutput],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Retry task-local construction when semantic constraints reject a seed."""

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
    raise RuntimeError("nonogram generation failed without a captured error")


def resolve_nonogram_public_branch(
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
        raise ValueError(f"unsupported nonogram public branch: {selected_branch}")
    return str(selected_branch), dict(branch_probabilities), task_params


def object_description_for_scene(scene_variant: str, mode: str) -> str:
    """Return concise scene wording for one nonogram visual treatment."""

    if str(mode) == "line_completion":
        base = "nonogram clue grid with one marked row and labeled row-strip options"
    else:
        base = "nonogram clue grid with labeled filled-grid options"
    if str(scene_variant) == "nonogram_card":
        return f"a card-style {base}"
    if str(scene_variant) == "nonogram_blueprint":
        return f"a blueprint-style {base}"
    return f"a {base}"


def prepare_nonogram_visual_case(
    *,
    dataset: NonogramDataset,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
    prompt_task_key: str,
    prompt_query_key: str,
    prompt_dynamic_slots: Mapping[str, Any],
    namespace: str,
) -> Dict[str, Any]:
    """Render a sampled dataset and matching prompt artifacts."""

    render_params = resolve_render_params(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    font_family = sample_nonogram_font(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
        namespace=namespace,
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    with temporary_default_font_family(str(font_family)):
        rendered_scene = render_nonogram_scene(
            background,
            scene_variant=str(scene_variant),
            mode=str(dataset.mode),
            display_grid=dataset.display_grid,
            row_clues=dataset.row_clues,
            col_clues=dataset.col_clues,
            render_params=render_params,
            marked_axis=dataset.marked_axis,
            marked_index=dataset.marked_index,
            option_specs=dataset.option_specs,
            show_empty_marks=bool(dataset.mode == "line_completion"),
            scene_style=scene_style,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    dynamic_slots = dict(prompt_dynamic_slots)
    dynamic_slots.setdefault(
        "object_description",
        object_description_for_scene(str(scene_variant), str(dataset.mode)),
    )
    prompt_defaults, prompt_artifacts = build_nonogram_prompt_artifacts(
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
        "font_meta": font_trace_record(str(font_family)),
        "prompt_defaults": dict(prompt_defaults),
        "prompt_artifacts": prompt_artifacts,
    }


def render_spec_from_visual(visual: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the common render-spec trace section from rendered visual metadata."""

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
            "font": dict(visual["font_meta"]),
        },
        "post_image_noise": dict(visual["post_noise_meta"]),
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "layout": "nonogram_clue_grid_with_visual_options",
        "text_style": {"font": dict(visual["font_meta"])},
        "unit_size_jitter": dict(render_params.unit_size_jitter),
    }


def nonogram_render_map(*, rendered_scene: Any, render_params: Any) -> Dict[str, Any]:
    """Build the common pixel render map for a rendered nonogram scene."""

    return with_puzzle_unit_size_jitter(
        {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "cell_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.cell_bbox_map.items()
            },
            "clue_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.clue_bbox_map.items()
            },
            "option_panel_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.option_panel_bbox_map.items()
            },
            "item_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.item_bbox_map.items()
            },
            "annotation_source": "option_panel_bboxes_px",
        },
        render_params.unit_size_jitter,
    )


def build_nonogram_task_output(
    *,
    dataset: NonogramDataset,
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
    """Assemble TaskOutput after the public task binds answer and annotation."""

    rendered_scene = visual["rendered_scene"]
    prompt_artifacts = visual["prompt_artifacts"]
    scene_variant = str(visual["scene_variant"])
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(public_query_id),
        params={
            "query_id_probabilities": dict(branch_probabilities),
            "prompt_query_key": str(prompt_query_key),
            "scene_variant": scene_variant,
            "grid_rows": len(dataset.grid),
            "grid_cols": len(dataset.grid[0]) if dataset.grid else 0,
            "option_count": int(dataset.option_count),
            **dict(semantic_params),
        },
    )
    render_map = nonogram_render_map(
        rendered_scene=rendered_scene,
        render_params=visual["render_params"],
    )
    dataset_record = asdict(dataset)
    trace_payload = build_trace_payload(
        scene_ir={
            "scene_kind": f"puzzle_nonogram_{scene_variant}",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "selected_branch": str(public_query_id),
                "semantic_rule": "binary_runs_must_match_row_and_column_nonogram_clues",
                "scene_variant": scene_variant,
                "answer_value": str(dataset.answer_value),
                "correct_option_panel_id": str(dataset.correct_option_panel_id),
                **dict(relation_fields),
            },
        },
        semantic_spec=query_spec,
        render_spec=render_spec_from_visual(visual),
        render_map=render_map,
        execution_trace={
            "query_id": str(public_query_id),
            "internal_query_id": str(prompt_query_key),
            "scene_id": SCENE_ID,
            "scene_variant": scene_variant,
            "question_format": str(prompt_query_key),
            "grid_rows": len(dataset.grid),
            "grid_cols": len(dataset.grid[0]) if dataset.grid else 0,
            "grid_rows_range": list(dataset.grid_rows_range),
            "grid_cols_range": list(dataset.grid_cols_range),
            "grid": dataset.grid,
            "display_grid": dataset.display_grid,
            "row_clues": dataset.row_clues,
            "col_clues": dataset.col_clues,
            "option_count": int(dataset.option_count),
            "option_specs": [dict(option) for option in dataset.option_specs],
            "answer_value": str(dataset.answer_value),
            "correct_option_panel_id": str(dataset.correct_option_panel_id),
            "correct_option_index": int(dataset.correct_option_index),
            "annotation_policy": "scalar_bbox_selected_visual_option_panel",
            "supporting_item_ids": [str(dataset.correct_option_panel_id)],
            "supporting_annotation_source": "option_panel_bboxes_px",
            "query_id_probabilities": dict(branch_probabilities),
            "dataset": dataset_record,
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


def build_nonogram_option_label_case(
    *,
    task_id: str,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    dataset_builder: Callable[..., tuple[NonogramDataset, dict[str, Any]]],
    prompt_dynamic_slots: Callable[[NonogramDataset], Mapping[str, Any]],
    validate_dataset: Callable[[NonogramDataset], None],
    relation_fields: Callable[[NonogramDataset], Mapping[str, Any]] | None = None,
) -> TaskOutput:
    """Run neutral nonogram option-label plumbing for a task-owned dataset."""

    selected_branch, branch_probabilities, task_params = resolve_nonogram_public_branch(
        instance_seed=int(instance_seed),
        params=params,
        task_id=str(task_id),
        namespace=str(namespace),
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    dataset, sampling_meta = dataset_builder(
        params=task_params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    validate_dataset(dataset)
    visual = prepare_nonogram_visual_case(
        dataset=dataset,
        params=task_params,
        rendering_defaults=rendering_defaults,
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        prompt_dynamic_slots=prompt_dynamic_slots(dataset),
        namespace=str(namespace),
    )
    annotation_gt, projected_annotation, witness_symbolic = option_panel_bbox(
        visual["rendered_scene"].item_bbox_map,
        dataset.correct_option_panel_id,
    )
    return build_nonogram_task_output(
        dataset=dataset,
        visual=visual,
        public_query_id=str(selected_branch),
        branch_probabilities=branch_probabilities,
        answer_gt=TypedValue(type="option_letter", value=str(dataset.answer_value)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        prompt_query_key=str(prompt_query_key),
        semantic_params={
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            **dict(sampling_meta),
        },
        relation_fields=dict(relation_fields(dataset) if relation_fields is not None else {}),
        execution_fields={
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            **dict(sampling_meta),
        },
    )


__all__ = [
    "build_nonogram_task_output",
    "build_nonogram_option_label_case",
    "object_description_for_scene",
    "prepare_nonogram_visual_case",
    "render_spec_from_visual",
    "resolve_nonogram_public_branch",
    "resolve_scene_variant",
    "retry_nonogram_generation",
]
