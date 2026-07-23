"""Scene-private lifecycle for public word-search puzzle tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
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
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.text_legibility import contrast_ratio

from .shared.defaults import resize_canvas_to_content, resolve_render_params
from .shared.output import build_trace_payload, json_ready
from .shared.prompts import render_word_search_prompt_artifacts
from .shared.rendering import render_word_search_scene
from .shared.state import DOMAIN, SCENE_ID, RenderedWordSearch, WordSearchDataset

_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15)
_SampleBuilder = Callable[
    [Mapping[str, Any], Mapping[str, Any], Any], WordSearchDataset
]
_OutputBinder = Callable[[WordSearchDataset, Mapping[str, Any]], "WordSearchBinding"]


@dataclass(frozen=True)
class WordSearchBinding:
    """Task-owned answer, annotation, and trace fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    semantic_params: Mapping[str, Any]
    execution_fields: Mapping[str, Any]


def make_word_search_binding(
    *,
    answer_type: str,
    answer_value: Any,
    annotation_result: tuple[TypedValue, Mapping[str, Any], Mapping[str, Any]],
    semantic_params: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
) -> WordSearchBinding:
    """Pack task-owned answer and annotation fields into a lifecycle binding."""

    annotation_gt, projected_annotation, witness_symbolic = annotation_result
    return WordSearchBinding(
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params=dict(semantic_params),
        execution_fields=dict(execution_fields),
    )


def run_word_search_single_query_case(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    namespace: str,
    prompt_task_key: str,
    prompt_query_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    sample_builder: _SampleBuilder,
    output_binder: _OutputBinder,
    attempt_limit: int,
) -> TaskOutput:
    """Run common word-search scene plumbing around task-owned callbacks."""

    generation_defaults, rendering_defaults, prompt_defaults = (
        load_scene_generation_rendering_prompt_defaults(
            DOMAIN,
            SCENE_ID,
            task_id=str(task_id),
        )
    )
    selected_query, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(task_id),
        namespace=f"{namespace}.query",
    )
    if str(selected_query) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported word-search query_id: {selected_query}")

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(attempt_limit))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        rng = spawn_rng(attempt_seed, f"{namespace}.sample")
        try:
            dataset = sample_builder(task_params, generation_defaults, rng)
            visual = prepare_word_search_visual_case(
                dataset=dataset,
                params=task_params,
                rendering_defaults=rendering_defaults,
                prompt_defaults=prompt_defaults,
                prompt_task_key=str(prompt_task_key),
                prompt_query_key=str(prompt_query_key),
                namespace=str(namespace),
                instance_seed=attempt_seed,
            )
            binding = output_binder(dataset, visual)
            return build_word_search_task_output(
                task_id=str(task_id),
                dataset=dataset,
                visual=visual,
                public_query_id=str(selected_query),
                branch_probabilities=dict(branch_probabilities),
                prompt_query_key=str(prompt_query_key),
                answer_gt=binding.answer_gt,
                annotation_gt=binding.annotation_gt,
                projected_annotation=binding.projected_annotation,
                witness_symbolic=binding.witness_symbolic,
                semantic_params=binding.semantic_params,
                execution_fields=binding.execution_fields,
                attempt_limit=int(attempt_limit),
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("word-search generation failed without a captured error")


def prepare_word_search_visual_case(
    *,
    dataset: WordSearchDataset,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    namespace: str,
    instance_seed: int,
) -> dict[str, Any]:
    """Resolve style, render image, and build prompt artifacts."""

    render_rng = spawn_rng(int(instance_seed), f"{namespace}.layout")
    render_params = resolve_render_params(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
    )
    render_params = resize_canvas_to_content(
        render_params,
        dataset=dataset,
        rng=render_rng,
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    render_params = replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        grid_fill_rgb=tuple(int(value) for value in scene_style.option_fill_rgb),
        header_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        grid_line_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        text_rgb=_contrast_text_rgb(
            scene_style.option_fill_rgb,
            scene_style.panel_fill_rgb,
        ),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
        option_fill_rgb=tuple(int(value) for value in scene_style.step_fill_rgb),
        option_border_rgb=tuple(int(value) for value in scene_style.mark_rgb),
        option_text_rgb=_contrast_text_rgb(
            scene_style.step_fill_rgb,
        ),
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_word_search_scene(
        background,
        dataset=dataset,
        render_params=render_params,
        rng=render_rng,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    prompt_artifacts = render_word_search_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=_prompt_dynamic_slots(dataset),
        instance_seed=int(instance_seed),
    )
    return {
        "image": image,
        "rendered_scene": rendered_scene,
        "render_params": render_params,
        "scene_style_meta": dict(scene_style_meta),
        "background_meta": dict(background_meta),
        "post_noise_meta": dict(post_noise_meta),
        "prompt_defaults": dict(prompt_defaults),
        "prompt_artifacts": prompt_artifacts,
    }


def build_word_search_task_output(
    *,
    task_id: str,
    dataset: WordSearchDataset,
    visual: Mapping[str, Any],
    public_query_id: str,
    branch_probabilities: Mapping[str, float],
    prompt_query_key: str,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    semantic_params: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
    attempt_limit: int,
) -> TaskOutput:
    """Assemble TaskOutput from task-owned answer and annotation binding."""

    rendered_scene: RenderedWordSearch = visual["rendered_scene"]
    render_params = visual["render_params"]
    prompt_artifacts = visual["prompt_artifacts"]
    query_params = {
        "query_id": str(public_query_id),
        "query_id_probabilities": dict(branch_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "scene_id": SCENE_ID,
        "grid_rows": int(dataset.rows),
        "grid_cols": int(dataset.cols),
        "grid_size_range": list(dataset.grid_size_range),
        "target_word": str(dataset.target_word),
        "target_letter": str(dataset.target_letter),
        "target_answer_support": list(dataset.answer_support),
        **dict(semantic_params),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(public_query_id),
        params=query_params,
    )
    annotation_source = str(
        execution_fields.get("supporting_annotation_source", "item_bboxes_px")
    )
    render_map = with_puzzle_unit_size_jitter(
        {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "cell_bboxes_px": dict(rendered_scene.cell_bbox_map),
            "cell_centers_px": dict(rendered_scene.cell_centers_px),
            "item_bboxes_px": dict(rendered_scene.item_bbox_map),
            "annotation_source": str(annotation_source),
            "layout_jitter": dict(rendered_scene.layout_jitter),
        },
        render_params.unit_size_jitter,
    )
    trace_payload = build_trace_payload(
        scene_ir={
            "scene_kind": f"puzzle_word_search_{dataset.scene_variant}",
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "answer_value": answer_gt.value,
        },
        query_spec=query_spec,
        render_spec={
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
            "scene_style": dict(visual["scene_style_meta"]),
            "background_style": dict(visual["background_meta"]),
            "post_image_noise": dict(visual["post_noise_meta"]),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "layout_jitter": dict(rendered_scene.layout_jitter),
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        render_map=render_map,
        execution_trace={
            **dict(query_params),
            "dataset": json_ready(dataset),
            "grid": [list(row) for row in dataset.grid],
            "word_bank": list(dataset.word_bank),
            "present_words": list(dataset.present_words),
            "placements": [_placement_record(item) for item in dataset.placements],
            "answer_value": answer_gt.value,
            "supporting_annotation_source": str(annotation_source),
            "task_id": str(task_id),
            "max_attempts": int(attempt_limit),
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


def _prompt_dynamic_slots(dataset: WordSearchDataset) -> dict[str, object]:
    """Return dynamic prompt slots shared by word-search prompt templates."""

    return {
        "target_word": str(dataset.target_word),
        "target_letter": str(dataset.target_letter),
        "word_bank_size": int(len(dataset.word_bank)),
    }


def _placement_record(placement) -> dict[str, Any]:
    """Serialize one word placement for trace metadata."""

    return {
        "word": str(placement.word),
        "start_row_1based": int(placement.row) + 1,
        "start_col_1based": int(placement.col) + 1,
        "direction": str(placement.direction),
        "cells": [[int(row), int(col)] for row, col in placement.cells],
    }


def _contrast_text_rgb(*surface_rgbs) -> tuple[int, int, int]:
    """Choose black or white text against the relevant visible surfaces."""

    surfaces = [tuple(int(value) for value in surface[:3]) for surface in surface_rgbs]
    candidates = ((12, 16, 24), (246, 248, 252))
    return max(
        candidates,
        key=lambda color: min(contrast_ratio(color, surface) for surface in surfaces)
        if surfaces
        else 1.0,
    )


__all__ = ["WordSearchBinding", "run_word_search_single_query_case"]
