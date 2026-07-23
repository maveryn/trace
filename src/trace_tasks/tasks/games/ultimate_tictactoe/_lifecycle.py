"""Private lifecycle plumbing for Ultimate Tic-Tac-Toe public tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, Callable

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import ultimate_bbox_annotation, ultimate_bbox_set_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, PROMPT_DEFAULTS
from .shared.prompts import build_ultimate_prompt_artifacts
from .shared.rendering import render_ultimate_tictactoe_scene
from .shared.sampling import sample_style_variant
from .shared.state import MACRO_LABELS, SCENE_ID, UltimateSample


PayloadPreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], str, Mapping[str, float], int],
    "UltimateTaskPayload",
]
AttemptBuilder = Callable[[Any], UltimateSample]
COUNT_JSON_EXAMPLES = (
    json.dumps({"annotation": [[110, 110, 260, 260], [450, 280, 600, 430]], "answer": 2}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
    json.dumps({"answer": 2}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
)
LABEL_JSON_EXAMPLES = (
    json.dumps(
        {
            "annotation": [410, 240, 470, 300],
            "answer": "C",
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ),
    json.dumps({"answer": "C"}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
)


@dataclass(frozen=True)
class UltimateTaskPayload:
    """Task-owned sample plus answer, prompt, and branch metadata."""

    sample: UltimateSample
    answer_gt: TypedValue
    prompt_key: str
    json_example: str
    json_example_answer_only: str
    branch_key: str
    branch_probabilities: Mapping[str, float]
    style_variant: str
    style_variant_probabilities: Mapping[str, float]
    annotation_kind: str
    semantic_params: Mapping[str, Any]


def sample_with_retry(
    *,
    public_id: str,
    namespace: str,
    instance_seed: int,
    max_attempts: int,
    build_attempt: AttemptBuilder,
) -> UltimateSample:
    """Retry task-owned construction with deterministic attempt namespaces."""

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{str(namespace)}.attempt.{int(attempt_index)}")
        try:
            return build_attempt(rng)
        except ValueError:
            continue
    raise RuntimeError(f"{public_id} failed to generate a valid Ultimate Tic-Tac-Toe board after {max_attempts} attempts")


def bind_ultimate_payload(
    *,
    sample: UltimateSample,
    answer_gt: TypedValue,
    prompt_key: str,
    branch_probabilities: Mapping[str, float],
    style_variant: str,
    style_variant_probabilities: Mapping[str, float],
    examples: tuple[str, str],
    semantic_params: Mapping[str, Any],
    annotation_kind: str = "bbox_set",
) -> UltimateTaskPayload:
    """Bind task-owned answer, prompt examples, and trace params to a sample."""

    return UltimateTaskPayload(
        sample=sample,
        answer_gt=answer_gt,
        prompt_key=str(prompt_key),
        json_example=str(examples[0]),
        json_example_answer_only=str(examples[1]),
        branch_key=str(prompt_key),
        branch_probabilities=dict(branch_probabilities),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_variant_probabilities),
        annotation_kind=str(annotation_kind),
        semantic_params=dict(semantic_params),
    )


def _small_board_trace(sample: UltimateSample) -> list[dict[str, Any]]:
    """Serialize the nine local boards for verifier and review traces."""

    return [
        {
            "label": MACRO_LABELS[int(index)],
            "status": str(local.status),
            "cells": [str(value) for value in local.cells],
            "winning_line": None if local.winning_line is None else [int(cell) + 1 for cell in local.winning_line],
        }
        for index, local in enumerate(sample.board)
    ]


def _trace_payload(
    *,
    payload: UltimateTaskPayload,
    prompt_query_spec: Mapping[str, Any],
    rendered_scene: Any,
    image_size: tuple[int, int],
    post_noise_meta: Mapping[str, Any],
    annotation_artifacts: Any,
) -> dict[str, Any]:
    """Build one trace payload after task code has fixed all semantics."""

    sample = payload.sample
    annotation_entity_ids = [str(entity_id) for entity_id in sample.annotation_entity_ids]
    return {
        "scene_ir": {
            "scene_kind": "games_ultimate_tictactoe_board",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "query_id": str(payload.branch_key),
                "style_variant": str(payload.style_variant),
                "annotation_entity_ids": list(annotation_entity_ids),
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "style_variant": str(payload.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.style_meta.get("panel_scene_style", {})),
            "ultimate_tictactoe_board_style": dict(rendered_scene.style_meta.get("ultimate_tictactoe_board_style", {})),
            "text_style": dict(rendered_scene.style_meta.get("text_style", {})),
            "effective_local_cell_size_px": rendered_scene.render_map.get("effective_local_cell_size_px"),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "query_id": str(payload.branch_key),
            "prompt_query_key": str(payload.prompt_key),
            "style_variant": str(payload.style_variant),
            "small_boards": _small_board_trace(sample),
            "target_answer": sample.target_answer,
            "highlighted_small_board": None
            if sample.highlighted_board_index is None
            else MACRO_LABELS[int(sample.highlighted_board_index)],
            "option_cells": [int(cell) + 1 for cell in sample.option_cells],
            "answer_cell": None if sample.answer_cell is None else int(sample.answer_cell) + 1,
            "support_cells": [int(cell) + 1 for cell in sample.support_cells],
            "answer": sample.answer,
            "annotation_entity_ids": list(annotation_entity_ids),
            "matching_small_boards": list(sample.metadata.get("matching_small_boards", [])),
            **dict(payload.semantic_params),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": list(annotation_entity_ids),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_scene.background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(PROMPT_DEFAULTS.get("bundle_id", ""))},
    }


def run_ultimate_lifecycle(
    *,
    public_id: str,
    supported_branches: tuple[str, ...],
    default_branch: str,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_payload: PayloadPreparer,
) -> TaskOutput:
    """Run branch/style selection before task-owned payload preparation."""

    branch_key, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_id),
        namespace=f"{str(namespace)}.branch",
    )
    style_variant, style_probabilities = sample_style_variant(
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=f"{str(namespace)}.style",
    )
    payload = prepare_payload(
        int(instance_seed),
        task_params,
        str(branch_key),
        branch_probabilities,
        str(style_variant),
        style_probabilities,
        int(max_attempts),
    )
    return build_ultimate_task_output(
        payload=payload,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=task_params,
    )


def build_ultimate_task_output(
    *,
    payload: UltimateTaskPayload,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> TaskOutput:
    """Render, prompt, project annotation, and assemble a public TaskOutput."""

    rendered_scene = render_ultimate_tictactoe_scene(
        sample=payload.sample,
        namespace=str(namespace),
        style_variant=str(payload.style_variant),
        instance_seed=int(instance_seed),
        params=params,
    )
    annotation_artifacts = (
        ultimate_bbox_annotation(rendered_scene, payload.sample.annotation_entity_ids[0])
        if str(payload.annotation_kind) == "bbox"
        else ultimate_bbox_set_annotation(rendered_scene, payload.sample.annotation_entity_ids)
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_artifacts = build_ultimate_prompt_artifacts(
        prompt_key=str(payload.prompt_key),
        json_example=str(payload.json_example),
        json_example_answer_only=str(payload.json_example_answer_only),
        instance_seed=int(instance_seed),
    )
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(payload.branch_key),
        params={
            "style_variant": str(payload.style_variant),
            "query_id_probabilities": dict(payload.branch_probabilities),
            "style_variant_probabilities": dict(payload.style_variant_probabilities),
            **dict(payload.sample.metadata),
            **dict(payload.semantic_params),
        },
    )
    trace_payload = _trace_payload(
        payload=payload,
        prompt_query_spec=prompt_query_spec,
        rendered_scene=rendered_scene,
        image_size=(int(image.size[0]), int(image.size[1])),
        post_noise_meta=post_noise_meta,
        annotation_artifacts=annotation_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=payload.answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(payload.branch_key),
    )
