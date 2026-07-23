"""Neutral visual/prompt preparation for polyomino assembly tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.sampling import uniform_choice_with_probabilities
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import selected_option_bbox_annotation
from .shared.defaults import apply_scene_palette, resolve_render_params
from .shared.defaults import resolve_scene_variant, resolve_total_cell_count
from .shared.output import build_trace_payload
from .shared.prompts import build_prompt
from .shared.rendering import render_polyomino_assembly_scene
from .shared.rules import can_two_pieces_tile_target, json_cells
from .shared.rules import pair_rotation_signature, reflection_signature
from .shared.rules import rotation_signature, shape_bbox_dims
from .shared.sampling import generation_bounds, option_index_for_label
from .shared.sampling import sample_connected_shape
from .shared.sampling import sample_pair_distractor, sample_shape_distractor
from .shared.sampling import sample_split_target
from .shared.sampling import serialize_pair_option_specs, serialize_shape_option_specs
from .shared.state import OPTION_LABELS, SCENE_ID, RenderedPolyominoAssemblyScene


_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

DatasetBuilder = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class PolyominoAssemblyArtifacts:
    """Prepared rendered scene and prompt artifacts for one generated sample."""

    image: Image.Image
    prompt: str
    prompt_variants: dict[str, str]
    prompt_meta: dict[str, Any]
    prompt_artifacts: PromptTraceArtifacts
    rendered_scene: RenderedPolyominoAssemblyScene
    render_params: Any
    background_meta: dict[str, Any]
    scene_style_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]


@dataclass(frozen=True)
class PolyominoAssemblyAxes:
    """Resolved fixed-query presentation and answer axes."""

    selected_query: str
    query_probabilities: dict[str, float]
    task_params: dict[str, Any]
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    total_cell_count: int
    total_cell_count_range: tuple[int, int]
    total_cell_count_probabilities: dict[str, float]
    answer_label: str
    answer_label_probabilities: dict[str, float]


def resolve_polyomino_assembly_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    task_id: str,
    namespace_base: str,
) -> PolyominoAssemblyAxes:
    """Resolve common fixed-query axes for a public task file."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=(SINGLE_QUERY_ID,),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(task_id),
        namespace=f"{str(namespace_base)}.query",
    )
    if str(selected_query) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query_id for {task_id}: {selected_query}")
    axes_rng = spawn_rng(int(instance_seed), f"{str(namespace_base)}.axes")
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        rng=axes_rng,
        params=task_params,
        generation_defaults=generation_defaults,
    )
    total_count, total_count_range, total_count_probabilities = resolve_total_cell_count(
        rng=axes_rng,
        params=task_params,
        generation_defaults=generation_defaults,
    )
    answer_rng = spawn_rng(int(instance_seed), f"{str(namespace_base)}.answer_label")
    answer_label, answer_label_probabilities = uniform_choice_with_probabilities(
        answer_rng,
        OPTION_LABELS,
    )
    return PolyominoAssemblyAxes(
        selected_query=str(selected_query),
        query_probabilities=dict(query_probabilities),
        task_params=dict(task_params),
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        total_cell_count=int(total_count),
        total_cell_count_range=(int(total_count_range[0]), int(total_count_range[1])),
        total_cell_count_probabilities=dict(total_count_probabilities),
        answer_label=str(answer_label),
        answer_label_probabilities=dict(answer_label_probabilities),
    )


def run_polyomino_assembly_public_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    task_id: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    namespace_base: str,
    dataset_builder: DatasetBuilder,
    question_format: str,
    view_family: str,
) -> TaskOutput:
    """Run retry and fixed-option plumbing around a task-local dataset builder."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            axes = resolve_polyomino_assembly_axes(
                instance_seed=int(attempt_seed),
                params=params,
                generation_defaults=generation_defaults,
                task_id=str(task_id),
                namespace_base=str(namespace_base),
            )
            dataset = dataset_builder(
                instance_seed=int(attempt_seed),
                params=axes.task_params,
                scene_variant=str(axes.scene_variant),
                total_cell_count=int(axes.total_cell_count),
                total_cell_count_range=axes.total_cell_count_range,
                total_cell_count_probabilities=axes.total_cell_count_probabilities,
                answer_label=str(axes.answer_label),
                answer_label_probabilities=axes.answer_label_probabilities,
            )
            return build_polyomino_assembly_task_output(
                instance_seed=int(attempt_seed),
                params=axes.task_params,
                rendering_defaults=rendering_defaults,
                prompt_defaults=prompt_defaults,
                dataset=dataset,
                axes=axes,
                prompt_task_key=str(prompt_task_key),
                prompt_query_key=str(prompt_query_key),
                namespace_base=str(namespace_base),
                question_format=str(question_format),
                view_family=str(view_family),
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{task_id} generation failed")


def prepare_polyomino_assembly_artifacts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    dataset: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    namespace_base: str,
) -> PolyominoAssemblyArtifacts:
    """Render one scene and build prompt artifacts from task-owned semantics."""

    render_params = resolve_render_params(
        params,
        rendering_defaults=rendering_defaults,
        instance_seed=int(instance_seed),
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace_base)}.background",
    )
    render_params = apply_scene_palette(render_params, scene_style)
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_polyomino_assembly_scene(
        background,
        scene_variant=str(dataset.get("scene_variant", "")),
        dataset=dataset,
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    prompt, prompt_variants, prompt_meta, prompt_artifacts = build_prompt(
        prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        instance_seed=int(instance_seed),
    )
    return PolyominoAssemblyArtifacts(
        image=image,
        prompt=prompt,
        prompt_variants=prompt_variants,
        prompt_meta=prompt_meta,
        prompt_artifacts=prompt_artifacts,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta=dict(background_meta),
        scene_style_meta=dict(scene_style_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def build_polyomino_assembly_task_output(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    dataset: Mapping[str, Any],
    axes: PolyominoAssemblyAxes,
    prompt_task_key: str,
    prompt_query_key: str,
    namespace_base: str,
    question_format: str,
    view_family: str,
) -> TaskOutput:
    """Assemble a fixed-option-label TaskOutput from task-owned dataset data."""

    artifacts = prepare_polyomino_assembly_artifacts(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        dataset=dataset,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        namespace_base=str(namespace_base),
    )
    annotation_artifacts = selected_option_bbox_annotation(
        artifacts.rendered_scene,
        str(dataset["correct_option_choice_id"]),
    )
    answer_gt = TypedValue(type="option_letter", value=str(dataset["answer_option_label"]))
    solver_trace = dataset.get("solver_trace", {})
    transform_policy = str(
        dataset.get(
            "transform_policy",
            solver_trace.get("transform_policy", "rotation_only_no_reflection")
            if isinstance(solver_trace, Mapping)
            else "rotation_only_no_reflection",
        )
    )
    task_fields = {
        "query_id": str(axes.selected_query),
        "query_id_probabilities": dict(axes.query_probabilities),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "answer_option_label_probabilities": dict(axes.answer_label_probabilities),
        "total_cell_count_range": list(axes.total_cell_count_range),
        "total_cell_count_probabilities": dict(axes.total_cell_count_probabilities),
        "transform_policy": str(transform_policy),
    }
    trace_payload = build_trace_payload(
        dataset=dataset,
        rendered_scene=artifacts.rendered_scene,
        render_params=artifacts.render_params,
        prompt_meta=artifacts.prompt_meta,
        task_fields=task_fields,
        background_meta=artifacts.background_meta,
        scene_style_meta=artifacts.scene_style_meta,
        post_noise_meta=artifacts.post_noise_meta,
        projected_annotation=annotation_artifacts.projected_annotation,
        question_format=str(question_format),
        view_family=str(view_family),
    )
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=artifacts.prompt_artifacts,
        query_id=str(axes.selected_query),
        params=task_fields,
    )
    return TaskOutput(
        prompt=artifacts.prompt,
        prompt_variants=artifacts.prompt_variants,
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=artifacts.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(axes.selected_query),
    )


def build_decomposition_dataset_payload(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace_base: str,
    scene_variant: str,
    total_cell_count: int,
    total_cell_count_range: tuple[int, int],
    total_cell_count_probabilities: Mapping[str, float],
    answer_label: str,
    answer_label_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Build a target shape and candidate two-piece decomposition options."""

    rng = spawn_rng(int(instance_seed), f"{str(namespace_base)}.dataset")
    bounds = generation_bounds(params, generation_defaults)
    split = sample_split_target(
        rng=rng,
        total_area=int(total_cell_count),
        min_piece_area=int(bounds["piece_cell_count_min"]),
        max_piece_area=int(bounds["piece_cell_count_max"]),
        max_dim=int(bounds["shape_bbox_max_dim"]),
        split_attempts=int(bounds["split_attempts"]),
    )
    target = split["target"]
    correct_pair = (split["piece_a"], split["piece_b"])
    correct_index = option_index_for_label(OPTION_LABELS, str(answer_label))
    option_pairs = _pair_options_with_unique_answer(
        rng=rng,
        target=target,
        correct_pair=correct_pair,
        correct_index=int(correct_index),
        max_dim=int(bounds["shape_bbox_max_dim"]),
    )
    option_specs = serialize_pair_option_specs(
        option_pairs=option_pairs,
        correct_option_index=int(correct_index),
    )
    return {
        "top_kind": "target",
        "scene_variant": str(scene_variant),
        "target_cells": json_cells(target),
        "option_specs": option_specs,
        "option_count": 4,
        "answer_option_label": str(answer_label),
        "answer_option_label_probabilities": dict(answer_label_probabilities),
        "correct_option_index": int(correct_index),
        "correct_option_choice_id": f"option_choice_{int(correct_index) + 1}",
        "total_cell_count": int(total_cell_count),
        "total_cell_count_range": list(total_cell_count_range),
        "total_cell_count_probabilities": dict(total_cell_count_probabilities),
        "solver_trace": {
            "transform_policy": "rotation_only_no_reflection",
            "target_cells": json_cells(target),
            "correct_piece_a_cells": json_cells(correct_pair[0]),
            "correct_piece_b_cells": json_cells(correct_pair[1]),
            "valid_option_choice_ids": [f"option_choice_{int(correct_index) + 1}"],
        },
    }


def build_composition_dataset_payload(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace_base: str,
    scene_variant: str,
    total_cell_count: int,
    total_cell_count_range: tuple[int, int],
    total_cell_count_probabilities: Mapping[str, float],
    answer_label: str,
    answer_label_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Build source pieces and candidate composite result options."""

    rng = spawn_rng(int(instance_seed), f"{str(namespace_base)}.dataset")
    bounds = generation_bounds(params, generation_defaults)
    split = sample_split_target(
        rng=rng,
        total_area=int(total_cell_count),
        min_piece_area=int(bounds["piece_cell_count_min"]),
        max_piece_area=int(bounds["piece_cell_count_max"]),
        max_dim=int(bounds["shape_bbox_max_dim"]),
        split_attempts=int(bounds["split_attempts"]),
    )
    source_pair = (split["piece_a"], split["piece_b"])
    correct_shape = split["target"]
    correct_index = option_index_for_label(OPTION_LABELS, str(answer_label))
    option_shapes = _shape_options_with_unique_answer(
        rng=rng,
        source_pair=source_pair,
        correct_shape=correct_shape,
        correct_index=int(correct_index),
        max_dim=int(bounds["shape_bbox_max_dim"]),
    )
    option_specs = serialize_shape_option_specs(
        option_shapes=option_shapes,
        correct_option_index=int(correct_index),
    )
    return {
        "top_kind": "source_pair",
        "scene_variant": str(scene_variant),
        "source_pieces": [
            {"piece_id": "piece_a", "cells": json_cells(source_pair[0])},
            {"piece_id": "piece_b", "cells": json_cells(source_pair[1])},
        ],
        "option_specs": option_specs,
        "option_count": 4,
        "answer_option_label": str(answer_label),
        "answer_option_label_probabilities": dict(answer_label_probabilities),
        "correct_option_index": int(correct_index),
        "correct_option_choice_id": f"option_choice_{int(correct_index) + 1}",
        "total_cell_count": int(total_cell_count),
        "total_cell_count_range": list(total_cell_count_range),
        "total_cell_count_probabilities": dict(total_cell_count_probabilities),
        "solver_trace": {
            "transform_policy": "rotation_only_no_reflection",
            "source_piece_a_cells": json_cells(source_pair[0]),
            "source_piece_b_cells": json_cells(source_pair[1]),
            "correct_shape_cells": json_cells(correct_shape),
            "valid_option_choice_ids": [f"option_choice_{int(correct_index) + 1}"],
        },
    }


def build_hole_fill_dataset_payload(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace_base: str,
    scene_variant: str,
    total_cell_count: int,
    total_cell_count_range: tuple[int, int],
    total_cell_count_probabilities: Mapping[str, float],
    answer_label: str,
    answer_label_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Build a polyomino board with one hole and candidate filler pieces."""

    rng = spawn_rng(int(instance_seed), f"{str(namespace_base)}.dataset")
    bounds = generation_bounds(params, generation_defaults)
    hole_shape = sample_connected_shape(
        rng=rng,
        area=int(total_cell_count),
        max_dim=int(bounds["shape_bbox_max_dim"]),
    )
    hole_width, hole_height = shape_bbox_dims(hole_shape)
    board_width = int(hole_width + 2)
    board_height = int(hole_height + 2)
    hole_cells = tuple((int(x + 1), int(y + 1)) for x, y in hole_shape)
    board_cells = tuple(
        (int(x), int(y))
        for y in range(int(board_height))
        for x in range(int(board_width))
        if (int(x), int(y)) not in set(hole_cells)
    )
    correct_index = option_index_for_label(OPTION_LABELS, str(answer_label))
    option_shapes = _hole_fill_options_with_unique_answer(
        rng=rng,
        hole_shape=hole_shape,
        correct_index=int(correct_index),
        max_dim=int(bounds["shape_bbox_max_dim"]),
    )
    option_specs = serialize_shape_option_specs(
        option_shapes=option_shapes,
        correct_option_index=int(correct_index),
    )
    return {
        "top_kind": "hole_board",
        "scene_variant": str(scene_variant),
        "board_width": int(board_width),
        "board_height": int(board_height),
        "board_cells": [[int(x), int(y)] for x, y in board_cells],
        "hole_cells": [[int(x), int(y)] for x, y in sorted(hole_cells)],
        "hole_shape_cells": json_cells(hole_shape),
        "option_specs": option_specs,
        "option_count": 4,
        "answer_option_label": str(answer_label),
        "answer_option_label_probabilities": dict(answer_label_probabilities),
        "correct_option_index": int(correct_index),
        "correct_option_choice_id": f"option_choice_{int(correct_index) + 1}",
        "total_cell_count": int(total_cell_count),
        "total_cell_count_range": list(total_cell_count_range),
        "total_cell_count_probabilities": dict(total_cell_count_probabilities),
        "transform_policy": "rotation_and_reflection_allowed",
        "solver_trace": {
            "transform_policy": "rotation_and_reflection_allowed",
            "board_width": int(board_width),
            "board_height": int(board_height),
            "hole_cells": [[int(x), int(y)] for x, y in sorted(hole_cells)],
            "hole_shape_cells": json_cells(hole_shape),
            "correct_piece_cells": json_cells(hole_shape),
            "valid_option_choice_ids": [f"option_choice_{int(correct_index) + 1}"],
        },
    }


def _pair_options_with_unique_answer(
    *,
    rng,
    target,
    correct_pair,
    correct_index: int,
    max_dim: int,
) -> list[tuple[Any, Any]]:
    """Construct four pair options and verify one exact target tiling answer."""

    option_pairs: list[tuple[Any, Any]] = [correct_pair]
    seen = {pair_rotation_signature(correct_pair[0], correct_pair[1])}
    total_area = int(len(correct_pair[0]) + len(correct_pair[1]))
    attempts = 0
    while len(option_pairs) < 4 and attempts < 400:
        attempts += 1
        pair = sample_pair_distractor(
            rng=rng,
            total_area=int(total_area),
            min_piece_area=2,
            max_piece_area=max(2, int(total_area) - 2),
            max_dim=int(max_dim),
        )
        signature = pair_rotation_signature(pair[0], pair[1])
        if signature in seen:
            continue
        seen.add(signature)
        option_pairs.append(pair)
    if len(option_pairs) < 4:
        raise RuntimeError("failed to construct enough pair distractors")
    rng.shuffle(option_pairs)
    correct_signature = pair_rotation_signature(correct_pair[0], correct_pair[1])
    option_pairs = [
        pair
        for pair in option_pairs
        if pair_rotation_signature(pair[0], pair[1]) != correct_signature
    ]
    option_pairs.insert(int(correct_index), correct_pair)
    option_pairs = option_pairs[:4]
    matches = [
        index
        for index, pair in enumerate(option_pairs)
        if can_two_pieces_tile_target(pair[0], pair[1], target)
    ]
    if matches != [int(correct_index)]:
        raise RuntimeError("decomposition options do not have a unique answer")
    return option_pairs


def _shape_options_with_unique_answer(
    *,
    rng,
    source_pair,
    correct_shape,
    correct_index: int,
    max_dim: int,
) -> list[Any]:
    """Construct four result-shape options and verify one exact composition."""

    source_a, source_b = source_pair
    option_shapes = [correct_shape]
    seen = {rotation_signature(correct_shape)}
    total_area = int(len(source_a) + len(source_b))
    preferred_dims = shape_bbox_dims(correct_shape)
    attempts = 0
    while len(option_shapes) < 4 and attempts < 400:
        attempts += 1
        candidate = sample_shape_distractor(
            rng=rng,
            total_area=int(total_area),
            max_dim=int(max_dim),
            preferred_dims=preferred_dims,
        )
        signature = rotation_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        option_shapes.append(candidate)
    if len(option_shapes) < 4:
        raise RuntimeError("failed to construct enough shape distractors")
    rng.shuffle(option_shapes)
    correct_signature = rotation_signature(correct_shape)
    option_shapes = [
        shape for shape in option_shapes if rotation_signature(shape) != correct_signature
    ]
    option_shapes.insert(int(correct_index), correct_shape)
    option_shapes = option_shapes[:4]
    matches = [
        index
        for index, shape in enumerate(option_shapes)
        if can_two_pieces_tile_target(source_a, source_b, shape)
    ]
    if matches != [int(correct_index)]:
        raise RuntimeError("composition options do not have a unique answer")
    return option_shapes


def _hole_fill_options_with_unique_answer(
    *,
    rng,
    hole_shape,
    correct_index: int,
    max_dim: int,
) -> list[Any]:
    """Construct four filler options and verify one reflection-aware answer."""

    option_shapes = [hole_shape]
    seen = {reflection_signature(hole_shape)}
    missing_area = int(len(hole_shape))
    preferred_dims = shape_bbox_dims(hole_shape)
    attempts = 0
    while len(option_shapes) < 4 and attempts < 500:
        attempts += 1
        candidate = sample_shape_distractor(
            rng=rng,
            total_area=int(missing_area),
            max_dim=int(max_dim),
            preferred_dims=preferred_dims,
        )
        signature = reflection_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        option_shapes.append(candidate)
    if len(option_shapes) < 4:
        raise RuntimeError("failed to construct enough hole-fill distractors")
    rng.shuffle(option_shapes)
    correct_signature = reflection_signature(hole_shape)
    option_shapes = [
        shape for shape in option_shapes if reflection_signature(shape) != correct_signature
    ]
    option_shapes.insert(int(correct_index), hole_shape)
    option_shapes = option_shapes[:4]
    matches = [
        index
        for index, shape in enumerate(option_shapes)
        if reflection_signature(shape) == correct_signature
    ]
    if matches != [int(correct_index)]:
        raise RuntimeError("hole-fill options do not have a unique answer")
    return option_shapes


__all__ = [
    "PolyominoAssemblyArtifacts",
    "PolyominoAssemblyAxes",
    "build_composition_dataset_payload",
    "build_decomposition_dataset_payload",
    "build_hole_fill_dataset_payload",
    "build_polyomino_assembly_task_output",
    "prepare_polyomino_assembly_artifacts",
    "resolve_polyomino_assembly_axes",
    "run_polyomino_assembly_public_task",
]
