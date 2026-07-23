"""Private neutral lifecycle plumbing for Tetris public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice

from .shared.annotations import tetris_annotation_bundle
from .shared.defaults import GEN_DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .shared.prompts import build_tetris_prompt_artifacts
from .shared.rendering import render_params, render_tetris_scene
from .shared.rules import placement_trace
from .shared.sampling import sample_scene_axes
from .shared.state import SCENE_ID, SceneAxes, TetrisSample


AttemptBuilder = Callable[[Any, SceneAxes], TetrisSample]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float], SceneAxes], "TetrisObjectivePlan"]


@dataclass(frozen=True)
class TetrisObjectivePlan:
    """Task-owned Tetris construction, prompt, render, and trace bindings."""

    attempt_namespace: str
    prompt_query_key: str
    answer_hint_key: str
    annotation_hint_key: str
    json_example: str
    json_example_answer_only: str
    render_mode: str
    query_params: Mapping[str, Any]
    dynamic_prompt_slots: Mapping[str, Any] = field(default_factory=dict)
    construct_attempt: AttemptBuilder | None = None


def _placement_payload(placement) -> dict[str, Any] | None:
    if placement is None:
        return None
    return placement_trace(placement)


def _option_payloads(sample: TetrisSample) -> list[dict[str, Any]]:
    """Serialize visual result-board options without owning objective logic."""

    return [
        {
            "label": str(option.label),
            "entity_id": str(option.entity_id),
            "board_rows": ["".join(row) for row in option.board],
            "placement": _placement_payload(option.placement),
            "is_answer": bool(option.is_answer),
            "metric_value": option.metric_value,
        }
        for option in sample.options
    ]


def _common_query_params(
    *,
    sample: TetrisSample,
    axes: SceneAxes,
    selected_query: str,
    prompt_query_key: str,
    query_probabilities: Mapping[str, float],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Return standard Tetris query metadata after the public task binds extras."""

    return {
        "prompt_query_key": str(prompt_query_key),
        "selected_public_query": str(selected_query),
        "public_query_probabilities": dict(query_probabilities),
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "option_count": int(axes.option_count),
        "option_count_probabilities": dict(axes.option_count_probabilities),
        "board_row_count": int(axes.board_rows),
        "board_row_probabilities": dict(axes.board_row_probabilities),
        "board_col_count": int(axes.board_cols),
        "board_col_probabilities": dict(axes.board_col_probabilities),
        "answer": sample.answer,
        **dict(sample.metadata),
        **dict(extra),
    }


def _trace_payload(
    *,
    sample: TetrisSample,
    rendered,
    axes: SceneAxes,
    selected_query: str,
    prompt_query_key: str,
    query_spec: Mapping[str, Any],
    annotation_bundle,
    prompt_bundle_id: str,
    post_noise_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble Tetris trace sections after task-specific answer binding."""

    return {
        "scene_ir": {
            "scene_kind": f"games_tetris_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "prompt_query_key": str(prompt_query_key),
                "selected_public_query": str(selected_query),
                "board_rows": ["".join(row) for row in sample.board],
                "piece": str(sample.piece),
                "answer": sample.answer,
                "annotation_entity_ids": [str(value) for value in sample.annotation_entity_ids],
            },
        },
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "layout_jitter": dict(rendered.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered.render_map.get("panel_scene_style", {})),
            "tetris_board_style": dict(rendered.render_map.get("tetris_board_style", {})),
            "text_style": dict(rendered.render_map.get("text_style", {})),
            "prompt_defaults_bundle_id": str(prompt_bundle_id),
        },
        "render_map": dict(rendered.render_map),
        "query_spec": dict(query_spec),
        "answer_gt": TypedValue(type=str(sample.answer_type), value=sample.answer).to_dict(),
        "annotation_gt": annotation_bundle.annotation_gt.to_dict(),
        "execution_trace": {
            "query_id": str(selected_query),
            "prompt_query_key": str(prompt_query_key),
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "board_rows": ["".join(row) for row in sample.board],
            "board_row_count": int(axes.board_rows),
            "board_col_count": int(axes.board_cols),
            "piece": str(sample.piece),
            "preview_orientation_index": int(sample.preview_orientation_index),
            "placement": _placement_payload(sample.placement),
            "falling_placement": _placement_payload(sample.falling_placement),
            "locked_cells": [[int(r), int(c)] for r, c in sample.outcome.locked_cells] if sample.outcome is not None else [],
            "cleared_rows": [int(row) for row in sample.outcome.cleared_rows] if sample.outcome is not None else [],
            "result_board_rows": ["".join(row) for row in sample.outcome.result_board] if sample.outcome is not None else [],
            "options": _option_payloads(sample),
            "answer": sample.answer,
            "annotation_kind": str(sample.annotation_kind),
            "annotation_entity_ids": [str(value) for value in sample.annotation_entity_ids],
            **dict(sample.metadata),
        },
        "witness_symbolic": dict(annotation_bundle.witness_symbolic),
        "projected_annotation": dict(annotation_bundle.projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


def resolve_tetris_integer_target(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: tuple[int, ...],
    namespace: str,
    balanced_flag_key: str = "balanced_target_answer_sampling",
    gen_defaults: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, float], tuple[int, ...]]:
    """Resolve one task-owned integer target and return its effective support."""

    effective_defaults = GEN_DEFAULTS if gen_defaults is None else dict(gen_defaults)
    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=effective_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return int(value), dict(probabilities), tuple(sorted(int(item) for item in probabilities))


def run_tetris_lifecycle(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run shared query selection, retry, rendering, prompt, and output plumbing."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = sample_scene_axes(
        namespace_key="games.tetris.scene_axes",
        instance_seed=int(instance_seed),
        params=task_params,
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query),
        dict(query_probabilities),
        axes,
    )
    if objective.construct_attempt is None:
        raise ValueError("Tetris objective plan must provide a construct_attempt callback")

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng, axes)
        except ValueError as exc:
            last_error = exc
            continue

        resolved_render_params = render_params(task_params, instance_seed=int(instance_seed))
        rendered = render_tetris_scene(
            sample=sample,
            render_mode=str(objective.render_mode),
            style_variant=str(axes.style_variant),
            params=resolved_render_params,
            instance_seed=int(instance_seed),
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        annotation_bundle = tetris_annotation_bundle(sample, rendered)
        prompt_defaults, prompt_artifacts = build_tetris_prompt_artifacts(
            prompt_query_key=str(objective.prompt_query_key),
            answer_hint_key=str(objective.answer_hint_key),
            annotation_hint_key=str(objective.annotation_hint_key),
            json_example=str(objective.json_example),
            json_example_answer_only=str(objective.json_example_answer_only),
            dynamic_slots=dict(objective.dynamic_prompt_slots),
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params=_common_query_params(
                sample=sample,
                axes=axes,
                selected_query=str(selected_query),
                prompt_query_key=str(objective.prompt_query_key),
                query_probabilities=dict(query_probabilities),
                extra=dict(objective.query_params),
            ),
        )
        trace_payload = _trace_payload(
            sample=sample,
            rendered=rendered,
            axes=axes,
            selected_query=str(selected_query),
            prompt_query_key=str(objective.prompt_query_key),
            query_spec=query_spec,
            annotation_bundle=annotation_bundle,
            prompt_bundle_id=str(prompt_defaults["bundle_id"]),
            post_noise_meta=post_noise_meta,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=TypedValue(type=str(sample.answer_type), value=sample.answer),
            annotation_gt=annotation_bundle.annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid Tetris scene: {last_error}") from last_error


__all__ = ["TetrisObjectivePlan", "resolve_tetris_integer_target", "run_tetris_lifecycle"]
