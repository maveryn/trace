"""Scene-private lifecycle plumbing for Star Battle public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.variant_sampling import resolve_variant

from .shared.annotations import bbox_artifacts, bbox_set_artifacts
from .shared.output import build_render_map, build_render_spec, build_trace_payload
from .shared.prompts import build_star_battle_prompt_artifacts
from .shared.rendering import apply_scene_style, render_star_battle_scene, resolve_render_params
from .shared.rules import candidate_key, cell_key, scope_item_ids
from .shared.state import (
    DOMAIN,
    RenderedStarBattleScene,
    SCENE_ID,
    SCENE_VARIANTS,
    StarBattleDataset,
    StarBattleRenderParams,
)


_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15)


@dataclass(frozen=True)
class StarBattleRenderContext:
    """Rendered Star Battle image and related metadata."""

    rendered_scene: RenderedStarBattleScene
    image: Image.Image
    render_params: StarBattleRenderParams
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    background_meta: Dict[str, Any]
    scene_style_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


@dataclass(frozen=True)
class BoundStarBattleOutput:
    """Task-owned answer, annotation, and trace fields after rendering."""

    prompt_query_key: str
    prompt_dynamic_slots: Dict[str, Any]
    query_params: Dict[str, Any]
    answer_gt: TypedValue
    annotation_artifacts: AnnotationArtifacts
    annotation_source: str
    scene_relations: Dict[str, Any]
    execution_trace: Dict[str, Any]
    witness_symbolic: Dict[str, Any]


DatasetBuilder = Callable[..., StarBattleDataset]
OutputBinder = Callable[..., BoundStarBattleOutput]


def select_scene_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    """Select one nonsemantic Star Battle visual variant."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    return resolve_variant(
        rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )


def render_star_battle_context(
    *,
    dataset: StarBattleDataset,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> StarBattleRenderContext:
    """Render a Star Battle dataset with shared puzzle treatment and noise."""

    render_params = resolve_render_params(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
    )
    render_params, background, background_meta, scene_style_meta = apply_scene_style(
        render_params=render_params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    rendered_scene = render_star_battle_scene(
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
    return StarBattleRenderContext(
        rendered_scene=rendered_scene,
        image=image,
        render_params=render_params,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        background_meta=dict(background_meta),
        scene_style_meta=dict(scene_style_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def build_common_query_params(
    *,
    dataset: StarBattleDataset,
    query_id: str,
    query_probabilities: Mapping[str, float],
    context: StarBattleRenderContext,
) -> Dict[str, Any]:
    """Build common query metadata shared by all Star Battle tasks."""

    params: Dict[str, Any] = {
        "query_id": str(query_id),
        "query_id_probabilities": dict(query_probabilities),
        "scene_id": SCENE_ID,
        "scene_variant": str(context.scene_variant),
        "scene_variant_probabilities": dict(context.scene_variant_probabilities),
        "grid_size": int(dataset.size),
        "grid_size_range": list(dataset.grid_size_range),
        "option_count": int(dataset.option_count),
        "target_answer_support": list(dataset.target_answer_support),
    }
    for key, value in (
        ("marked_region_index", dataset.marked_region_index),
        ("marked_row_index", dataset.marked_row_index),
        ("marked_col_index", dataset.marked_col_index),
        ("target_count_range", dataset.target_count_range),
        ("correct_option_index", dataset.correct_option_index),
    ):
        if value is not None:
            params[key] = list(value) if isinstance(value, tuple) else value
    return params


def dataset_trace_fields(dataset: StarBattleDataset) -> Dict[str, Any]:
    """Return JSON-ready symbolic Star Battle fields for execution trace."""

    fields: Dict[str, Any] = {
        "solution_stars": [[int(r), int(c)] for r, c in dataset.solution_stars],
        "visible_stars": [[int(r), int(c)] for r, c in dataset.visible_stars],
        "region_grid": [[int(value) for value in row] for row in dataset.region_grid],
        "regions": {
            str(key): [[int(r), int(c)] for r, c in cells]
            for key, cells in dict(dataset.regions).items()
        },
        "candidate_specs": [
            {
                "label": str(spec.label),
                "row": int(spec.row),
                "col": int(spec.col),
                "is_correct": bool(spec.is_correct),
                "is_legal": bool(spec.is_legal),
            }
            for spec in dataset.candidate_specs
        ],
        "legal_cells": [[int(r), int(c)] for r, c in dataset.legal_cells],
        "scope_cells": [[int(r), int(c)] for r, c in dataset.scope_cells],
        "scoped_legal_cells": [[int(r), int(c)] for r, c in dataset.scoped_legal_cells],
        "answer_value": dataset.answer_value,
    }
    if dataset.correct_cell is not None:
        fields["correct_cell"] = [int(value) for value in dataset.correct_cell]
    return fields


def bind_valid_cell_label_output(
    *,
    dataset: StarBattleDataset,
    context: StarBattleRenderContext,
    query_id: str,
    query_probabilities: Mapping[str, float],
    scope_kind: str,
) -> BoundStarBattleOutput:
    """Bind answer and scalar selected-cell bbox for a valid-cell option task."""

    selected_item_id = candidate_key(str(dataset.answer_value))
    annotation = bbox_artifacts(context.rendered_scene.item_bbox_map, selected_item_id)
    scope_ids = scope_item_ids(
        scope_kind=str(scope_kind),
        marked_region_index=dataset.marked_region_index,
        marked_row_index=dataset.marked_row_index,
        marked_col_index=dataset.marked_col_index,
    )
    query_params = build_common_query_params(
        dataset=dataset,
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        context=context,
    )
    query_params["scope_kind"] = str(scope_kind)
    return BoundStarBattleOutput(
        prompt_query_key=str(query_id),
        prompt_dynamic_slots={},
        query_params=query_params,
        answer_gt=TypedValue(type="option_letter", value=str(dataset.answer_value)),
        annotation_artifacts=annotation,
        annotation_source="item_bboxes_px",
        scene_relations={
            "query_id": str(query_id),
            "scene_id": SCENE_ID,
            "scene_variant": str(context.scene_variant),
            "answer_value": str(dataset.answer_value),
            "scope_kind": str(scope_kind),
        },
        execution_trace={
            **dict(query_params),
            **dataset_trace_fields(dataset),
            "supporting_item_ids": [selected_item_id, *scope_ids],
            "question_format": str(query_id),
        },
        witness_symbolic={"type": "bbox", "value": list(annotation.value)},
    )


def bind_remaining_count_output(
    *,
    dataset: StarBattleDataset,
    context: StarBattleRenderContext,
    query_id: str,
    query_probabilities: Mapping[str, float],
    scope_kind: str,
) -> BoundStarBattleOutput:
    """Bind answer and bbox-set annotation for counted legal cells."""

    counted_cell_ids = [cell_key(tuple(cell)) for cell in dataset.scoped_legal_cells]
    annotation = bbox_set_artifacts(context.rendered_scene.item_bbox_map, counted_cell_ids)
    scope_ids = scope_item_ids(
        scope_kind=str(scope_kind),
        marked_region_index=dataset.marked_region_index,
        marked_row_index=dataset.marked_row_index,
        marked_col_index=dataset.marked_col_index,
    )
    query_params = build_common_query_params(
        dataset=dataset,
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        context=context,
    )
    query_params["scope_kind"] = str(scope_kind)
    prompt_dynamic_slots: Dict[str, Any] = {}
    if str(scope_kind) == "marked_row":
        prompt_dynamic_slots["row_index"] = int(dataset.marked_row_index) + 1
    elif str(scope_kind) == "marked_column":
        prompt_dynamic_slots["column_index"] = int(dataset.marked_col_index) + 1
    return BoundStarBattleOutput(
        prompt_query_key=str(query_id),
        prompt_dynamic_slots=prompt_dynamic_slots,
        query_params=query_params,
        answer_gt=TypedValue(type="integer", value=int(dataset.answer_value)),
        annotation_artifacts=annotation,
        annotation_source="item_bboxes_px",
        scene_relations={
            "query_id": str(query_id),
            "scene_id": SCENE_ID,
            "scene_variant": str(context.scene_variant),
            "answer_value": int(dataset.answer_value),
            "scope_kind": str(scope_kind),
        },
        execution_trace={
            **dict(query_params),
            **dataset_trace_fields(dataset),
            "supporting_item_ids": [*scope_ids, *counted_cell_ids],
            "counted_cell_ids": list(counted_cell_ids),
            "question_format": str(query_id),
        },
        witness_symbolic={"type": "bbox_set", "value": [list(bbox) for bbox in annotation.value]},
    )


def finalize_star_battle_output(
    *,
    context: StarBattleRenderContext,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    prompt_dynamic_slots: Mapping[str, Any],
    query_id: str,
    query_params: Mapping[str, Any],
    answer_gt: TypedValue,
    annotation_artifacts: AnnotationArtifacts,
    annotation_source: str,
    scene_relations: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    instance_seed: int,
) -> TaskOutput:
    """Build prompt artifacts, trace payload, and the final TaskOutput."""

    prompt_defaults_resolved, prompt_artifacts = build_star_battle_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(prompt_dynamic_slots),
        instance_seed=int(instance_seed),
    )
    trace_payload = build_trace_payload(
        scene_ir={
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in context.rendered_scene.entities],
            "relations": dict(scene_relations),
        },
        query_spec=build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params=dict(query_params),
        ),
        render_spec=build_render_spec(
            rendered_scene=context.rendered_scene,
            render_params=context.render_params,
            scene_variant=str(context.scene_variant),
            background_meta=context.background_meta,
            scene_style_meta=context.scene_style_meta,
            post_noise_meta=context.post_noise_meta,
        ),
        render_map=build_render_map(
            rendered_scene=context.rendered_scene,
            render_params=context.render_params,
            annotation_source=str(annotation_source),
        ),
        execution_trace=dict(execution_trace),
        witness_symbolic=dict(witness_symbolic),
        projected_annotation=dict(annotation_artifacts.projected_annotation),
        answer_gt=answer_gt.to_dict(),
        annotation_gt=annotation_artifacts.annotation_gt.to_dict(),
        prompt_defaults=prompt_defaults_resolved,
        prompt_artifacts=prompt_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=context.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def run_star_battle_public_task(
    *,
    task_id: str,
    supported_query_ids: Sequence[str],
    prompt_task_key: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    dataset_builder: DatasetBuilder,
    output_binder: OutputBinder,
) -> TaskOutput:
    """Run common lifecycle steps around task-owned dataset and output callbacks."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    scene_variant, scene_variant_probabilities = select_scene_variant(
        instance_seed=int(instance_seed),
        params=task_params,
        generation_defaults=generation_defaults,
        namespace=str(task_id),
    )
    dataset = None
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            dataset = dataset_builder(
                query_id=str(query_id),
                params=task_params,
                generation_defaults=generation_defaults,
                namespace=str(task_id),
                instance_seed=int(instance_seed) + int(attempt_index),
            )
            break
        except (RuntimeError, ValueError) as exc:
            last_error = exc
    if dataset is None:
        raise RuntimeError(f"failed to generate Star Battle task {task_id}") from last_error

    context = render_star_battle_context(
        dataset=dataset,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_variant_probabilities,
        instance_seed=int(instance_seed),
        params=task_params,
        rendering_defaults=rendering_defaults,
        namespace=str(task_id),
    )
    bound = output_binder(
        dataset=dataset,
        context=context,
        query_id=str(query_id),
        query_probabilities=dict(query_probabilities),
    )
    return finalize_star_battle_output(
        context=context,
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(bound.prompt_query_key),
        prompt_dynamic_slots=bound.prompt_dynamic_slots,
        query_id=str(query_id),
        query_params=bound.query_params,
        answer_gt=bound.answer_gt,
        annotation_artifacts=bound.annotation_artifacts,
        annotation_source=str(bound.annotation_source),
        scene_relations=bound.scene_relations,
        execution_trace=bound.execution_trace,
        witness_symbolic=bound.witness_symbolic,
        instance_seed=int(instance_seed),
    )


__all__ = [
    "BoundStarBattleOutput",
    "StarBattleRenderContext",
    "bind_remaining_count_output",
    "bind_valid_cell_label_output",
    "run_star_battle_public_task",
]
