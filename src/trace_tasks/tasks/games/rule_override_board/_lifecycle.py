"""Neutral lifecycle plumbing for rule-override board public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.annotations import board_bbox_set_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS
from .shared.prompts import build_rule_override_prompt_artifacts
from .shared.rendering import make_rule_override_background, render_rule_override_scene, resolve_render_params
from .shared.rules import opponent
from .shared.state import RuleOverrideAxes, RuleOverrideSceneSample, SCENE_ID


@dataclass(frozen=True)
class ObjectiveRuleOverridePlan:
    """Task-owned objective hooks consumed by neutral scene lifecycle code."""

    axes: RuleOverrideAxes
    attempt_namespace: str
    prompt_query_key: str
    query_params: Mapping[str, Any]
    construct_attempt: Callable[[Any], RuleOverrideSceneSample]


def _board_trace_rows(sample: RuleOverrideSceneSample) -> list[dict[str, Any]]:
    """Serialize mini-board symbolic state for verifier trace payloads."""

    return [
        {
            "board_id": str(board.board_id),
            "label": str(board.label),
            "cells": [list(row) for row in board.cells],
            "counted": bool(board.counted),
            "result_for_target_player": str(board.result),
            "target_stat": int(board.target_stat),
            "opponent_stat": int(board.opponent_stat),
        }
        for board in sample.boards
    ]


def _build_trace_payload(
    *,
    sample: RuleOverrideSceneSample,
    axes: RuleOverrideAxes,
    rendered: Any,
    annotation: Any,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    prompt_defaults_resolved: Mapping[str, Any],
    prompt_artifacts: Any,
    render_params: Any,
    panel_style_meta: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    query_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Build task trace payload in private lifecycle, not shared primitives."""

    payload = {
        "scene_ir": {
            "scene_kind": "games_rule_override_board",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": str(selected_query_id),
                "board_family": str(sample.board_family),
                "board_style": str(sample.board_style),
                "target_player": str(sample.target_player),
                "rule_text": str(sample.rule_text),
                "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
            },
        },
        "query_spec": {
            "query_id": str(selected_query_id),
            "template_id": str(prompt_defaults_resolved["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "query_id": str(selected_query_id),
                "board_family": str(sample.board_family),
                "board_style": str(sample.board_style),
                "target_player": str(sample.target_player),
                "board_count": int(len(sample.boards)),
                "board_size": int(sample.board_size),
                "target_answer": int(sample.answer),
                "query_id_probabilities": dict(query_probabilities),
                "board_style_probabilities": dict(axes.board_style_probabilities),
                "target_player_probabilities": dict(axes.target_player_probabilities),
                "board_count_probabilities": dict(axes.board_count_probabilities),
                "board_size_probabilities": dict(axes.board_size_probabilities),
                "target_answer_probabilities": dict(axes.target_answer_probabilities),
                **dict(query_params),
            },
        },
        "render_spec": {
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "board_style": str(sample.board_style),
            "layout_jitter": dict(rendered.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(panel_style_meta),
            "text_style": dict(rendered.render_map.get("text_style", {})),
            "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "query_id": str(selected_query_id),
            "board_family": str(sample.board_family),
            "board_style": str(sample.board_style),
            "target_player": str(sample.target_player),
            "opponent_player": opponent(str(sample.target_player)),
            "rule_text": str(sample.rule_text),
            "answer": int(sample.answer),
            "boards": _board_trace_rows(sample),
            "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
        },
        "projected_annotation": dict(annotation.projected_annotation),
        "background": dict(background_meta),
        "panel_scene_style": dict(panel_style_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    return payload


def run_rule_override_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: Sequence[str],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: Callable[[int, Mapping[str, Any], Mapping[str, float], str], ObjectiveRuleOverridePlan],
) -> TaskOutput:
    """Run shared rendering/prompt/output steps after task-owned objective setup."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    objective = prepare_objective(int(instance_seed), task_params, query_probabilities, str(selected_query_id))
    sample: RuleOverrideSceneSample | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng)
        except ValueError:
            continue
        break
    if sample is None:
        raise RuntimeError(f"{task_id} failed to generate after {max_attempts} attempts")

    render_params = resolve_render_params(
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        axes=objective.axes,
    )
    background, background_meta, panel_style, panel_style_meta = make_rule_override_background(
        instance_seed=int(instance_seed),
        gen_defaults=gen_defaults,
        render_params=render_params,
    )
    rendered = render_rule_override_scene(sample=sample, params=render_params, panel_style=panel_style, background=background)
    annotation = board_bbox_set_annotation(rendered, sample.annotation_entity_ids)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_resolved, prompt_artifacts = build_rule_override_prompt_artifacts(
        domain=str(domain),
        prompt_query_key=str(objective.prompt_query_key),
        prompt_defaults=prompt_defaults,
        target_player=str(sample.target_player),
        rule_text=str(sample.rule_text),
        instance_seed=int(instance_seed),
    )
    trace_payload = _build_trace_payload(
        sample=sample,
        axes=objective.axes,
        rendered=rendered,
        annotation=annotation,
        selected_query_id=str(selected_query_id),
        query_probabilities=query_probabilities,
        prompt_defaults_resolved=prompt_defaults_resolved,
        prompt_artifacts=prompt_artifacts,
        render_params=render_params,
        panel_style_meta=panel_style_meta,
        background_meta=background_meta,
        post_noise_meta=post_noise_meta,
        query_params=objective.query_params,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        annotation_gt=annotation.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


__all__ = ["ObjectiveRuleOverridePlan", "run_rule_override_lifecycle"]
