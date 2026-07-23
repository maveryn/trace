"""Scene-private lifecycle plumbing for Sudoku public task files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import (
    bbox_for_coord,
    bbox_payload,
)
from .shared.output import build_sudoku_trace_payload, text_style_metadata
from .shared.prompts import (
    object_description_for_scene_variant,
    render_sudoku_prompt_artifacts,
    unit_scope_text,
)
from .shared.rendering import RenderedSudokuScene, render_sudoku_visual_artifacts
from .shared.sampling import (
    SudokuAxes,
    SudokuDefaults,
    build_sudoku_solution,
    resolve_sudoku_answer_label,
    resolve_sudoku_axes,
    resolve_sudoku_render_params,
    resolve_sudoku_scene_variant,
    resolve_sudoku_style_variant,
    resolve_sudoku_target_digit,
    resolve_sudoku_unit_type,
)
from .shared.state import Board, SCENE_ID, SudokuSample


@dataclass(frozen=True)
class SudokuAnnotationBinding:
    """Task-owned annotation projection result consumed by lifecycle plumbing."""

    annotation_kind: str
    annotation_value: Any
    entity_ids: Any


@dataclass(frozen=True)
class SudokuTaskRuntime:
    """Public task metadata required by neutral Sudoku lifecycle plumbing."""

    source_id: str
    support_key: str
    include_unit_type: bool
    prompt_query_key: str
    attempt_namespace: str


@dataclass(frozen=True)
class SudokuOptionAxes:
    """Resolved axes for in-grid option-letter Sudoku tasks."""

    scene_variant: str
    style_variant: str
    option_labels: tuple[str, ...]
    answer_label: str
    answer_label_probabilities: dict[str, float]
    scene_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]
    target_digit: int | None = None
    target_digit_support: tuple[int, ...] = ()
    target_digit_probabilities: dict[str, float] | None = None
    unit_type: str | None = None
    unit_type_probabilities: dict[str, float] | None = None


SampleBuilder = Callable[..., SudokuSample]


def _bind_sudoku_annotation(
    rendered_scene: RenderedSudokuScene,
    sample: SudokuSample,
) -> SudokuAnnotationBinding:
    """Project sample-selected Sudoku cell witnesses into public annotation."""

    cell_bboxes = rendered_scene.render_map["cell_bboxes_px"]
    if len(sample.annotation_coords) == 1:
        projected, entity_ids = bbox_for_coord(
            cell_bboxes,
            sample.annotation_coords[0],
        )
        return SudokuAnnotationBinding(
            annotation_kind="bbox",
            annotation_value=projected["bbox"],
            entity_ids=entity_ids,
        )
    if sample.highlighted_unit_type is None or sample.highlighted_unit_index is None:
        raise RuntimeError(
            "Sudoku scalar annotation requires a marked cell or highlighted unit"
        )
    highlighted_unit_bbox = rendered_scene.render_map.get("highlighted_unit_bbox_px")
    if highlighted_unit_bbox is None:
        raise RuntimeError("missing highlighted Sudoku unit bbox for annotation")
    projected = bbox_payload(highlighted_unit_bbox)
    entity_ids = [
        str(item) for item in rendered_scene.render_map["highlighted_cell_ids"]
    ]
    return SudokuAnnotationBinding(
        annotation_kind="bbox",
        annotation_value=projected["bbox"],
        entity_ids=entity_ids,
    )


def run_sudoku_lifecycle(
    *,
    source_id: str,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: SudokuDefaults,
    support_key: str,
    fallback_support: Sequence[int],
    include_unit_type: bool,
    prompt_query_key: str,
    prompt_required_keys: Sequence[str],
    attempt_namespace: str,
    build_sample: SampleBuilder,
    noise_defaults: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
) -> TaskOutput:
    """Run neutral Sudoku rendering/retry plumbing around task-owned hooks."""

    axes = resolve_sudoku_axes(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=str(attempt_namespace),
        support_key=str(support_key),
        fallback_support=fallback_support,
        include_unit_type=bool(include_unit_type),
    )
    render_params = resolve_sudoku_render_params(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        defaults=defaults,
    )

    sample: SudokuSample | None = None
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            solution = build_sudoku_solution(rng)
            sample = build_sample(
                rng=rng,
                solution=solution,
                axes=axes,
                defaults=defaults,
            )
            break
        except ValueError as exc:
            last_error = exc
    if sample is None:
        raise RuntimeError(
            f"{source_id} failed to generate a valid Sudoku sample after "
            f"{max_attempts} attempts"
        ) from last_error

    visual = render_sudoku_visual_artifacts(
        sample=sample,
        style_variant=str(axes.style_variant),
        render_params=render_params,
        instance_seed=int(instance_seed),
        params=dict(params),
        noise_defaults=dict(noise_defaults),
    )
    annotation = _bind_sudoku_annotation(visual.rendered_scene, sample)
    prompt_defaults_resolved = required_group_defaults(
        prompt_defaults,
        tuple(str(key) for key in prompt_required_keys),
        context=f"prompt defaults for {source_id}",
    )
    object_description = object_description_for_scene_variant(
        prompt_defaults_resolved,
        str(axes.scene_variant),
    )
    prompt_artifacts = render_sudoku_prompt_artifacts(
        prompt_defaults=prompt_defaults_resolved,
        prompt_query_key=str(prompt_query_key),
        object_description=str(object_description),
        unit_scope_text=unit_scope_text(
            prompt_defaults_resolved,
            sample.highlighted_unit_type,
        ),
        instance_seed=int(instance_seed),
    )
    query_params = {
        "query_id_probabilities": dict(query_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "target_answer": int(sample.answer),
        "target_answer_support": [int(value) for value in axes.target_answer_support],
        "target_answer_probabilities": dict(axes.target_answer_probabilities),
        "visible_count": int(sample.visible_count),
    }
    if sample.highlighted_unit_type is not None:
        query_params.update(
            {
                "unit_type": sample.highlighted_unit_type,
                "unit_index": sample.highlighted_unit_index,
                "unit_type_probabilities": dict(axes.unit_type_probabilities or {}),
            }
        )
    prompt_spec_payload = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=query_params,
    )
    trace_payload = build_sudoku_trace_payload(
        sample=sample,
        rendered_scene=visual.rendered_scene,
        image=visual.image,
        prompt_artifacts=prompt_artifacts,
        prompt_spec_payload=prompt_spec_payload,
        execution_fields={"query_id": str(selected_query_id)},
        annotation_type=str(annotation.annotation_kind),
        annotation_value=annotation.annotation_value,
        annotation_entity_ids=annotation.entity_ids,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        panel_style_meta=visual.panel_style_meta,
        text_style_meta=text_style_metadata(str(render_params.font_family)),
        background_meta=visual.background_meta,
        post_noise_meta=visual.post_noise_meta,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        annotation_gt=TypedValue(
            type=str(annotation.annotation_kind),
            value=annotation.annotation_value,
        ),
        image=visual.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


def _resolve_sudoku_option_axes(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: SudokuDefaults,
    instance_seed: int,
    namespace_root: str,
    include_unit_type: bool,
    target_digit_support_key: str | None,
) -> SudokuOptionAxes:
    """Resolve scene/style plus option-letter axes for Sudoku option tasks."""

    scene_variant, scene_probs = resolve_sudoku_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=str(namespace_root),
    )
    style_variant, style_probs = resolve_sudoku_style_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=str(namespace_root),
    )
    answer_label, option_labels, answer_probs = resolve_sudoku_answer_label(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=str(namespace_root),
        fallback=defaults.option_label_support,
    )
    target_digit = None
    target_digit_support: tuple[int, ...] = ()
    target_digit_probabilities = None
    if target_digit_support_key is not None:
        digit_fallback = getattr(defaults, str(target_digit_support_key))
        target_digit, target_digit_support, target_digit_probabilities = (
            resolve_sudoku_target_digit(
                params,
                gen_defaults=gen_defaults,
                instance_seed=int(instance_seed),
                namespace_root=str(namespace_root),
                support_key=str(target_digit_support_key),
                fallback_support=digit_fallback,
            )
        )

    unit_type = None
    unit_probs = None
    if include_unit_type:
        unit_type, unit_probs = resolve_sudoku_unit_type(
            params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            namespace_root=str(namespace_root),
        )

    return SudokuOptionAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        option_labels=tuple(str(label) for label in option_labels),
        answer_label=str(answer_label),
        answer_label_probabilities=dict(answer_probs),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
        target_digit=int(target_digit) if target_digit is not None else None,
        target_digit_support=tuple(int(value) for value in target_digit_support),
        target_digit_probabilities=(
            dict(target_digit_probabilities)
            if target_digit_probabilities is not None
            else None
        ),
        unit_type=str(unit_type) if unit_type is not None else None,
        unit_type_probabilities=dict(unit_probs) if unit_probs is not None else None,
    )


def run_sudoku_option_lifecycle(
    *,
    source_id: str,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    defaults: SudokuDefaults,
    include_unit_type: bool,
    target_digit_support_key: str | None,
    prompt_query_key: str,
    prompt_required_keys: Sequence[str],
    attempt_namespace: str,
    build_sample: SampleBuilder,
    noise_defaults: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
) -> TaskOutput:
    """Run Sudoku rendering/retry plumbing for option-letter tasks."""

    axes = _resolve_sudoku_option_axes(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        instance_seed=int(instance_seed),
        namespace_root=str(attempt_namespace),
        include_unit_type=bool(include_unit_type),
        target_digit_support_key=target_digit_support_key,
    )
    render_params = resolve_sudoku_render_params(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        defaults=defaults,
    )

    sample: SudokuSample | None = None
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            solution = build_sudoku_solution(rng)
            sample = build_sample(
                rng=rng,
                solution=solution,
                axes=axes,
                defaults=defaults,
            )
            if str(sample.answer) != str(axes.answer_label):
                raise ValueError(
                    "constructed Sudoku option sample has wrong answer label"
                )
            break
        except ValueError as exc:
            last_error = exc
    if sample is None:
        raise RuntimeError(
            f"{source_id} failed to generate a valid Sudoku option sample after "
            f"{max_attempts} attempts"
        ) from last_error

    visual = render_sudoku_visual_artifacts(
        sample=sample,
        style_variant=str(axes.style_variant),
        render_params=render_params,
        instance_seed=int(instance_seed),
        params=dict(params),
        noise_defaults=dict(noise_defaults),
    )
    annotation = _bind_sudoku_annotation(visual.rendered_scene, sample)
    prompt_defaults_resolved = required_group_defaults(
        prompt_defaults,
        tuple(str(key) for key in prompt_required_keys),
        context=f"prompt defaults for {source_id}",
    )
    object_description = object_description_for_scene_variant(
        prompt_defaults_resolved,
        str(axes.scene_variant),
    )
    prompt_artifacts = render_sudoku_prompt_artifacts(
        prompt_defaults=prompt_defaults_resolved,
        prompt_query_key=str(prompt_query_key),
        object_description=str(object_description),
        unit_scope_text=unit_scope_text(
            prompt_defaults_resolved,
            sample.highlighted_unit_type,
        ),
        target_digit=str(sample.target_digit or axes.target_digit or ""),
        instance_seed=int(instance_seed),
    )
    query_params = {
        "query_id_probabilities": dict(query_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "target_answer": str(sample.answer),
        "target_answer_support": [str(label) for label in axes.option_labels],
        "target_answer_probabilities": dict(axes.answer_label_probabilities),
        "option_count": len(axes.option_labels),
        "option_labels": [str(label) for label in axes.option_labels],
        "visible_count": int(sample.visible_count),
    }
    if sample.target_digit is not None:
        query_params.update(
            {
                "target_digit": int(sample.target_digit),
                "target_digit_support": [
                    int(value) for value in axes.target_digit_support
                ],
                "target_digit_probabilities": dict(
                    axes.target_digit_probabilities or {}
                ),
            }
        )
    if sample.highlighted_unit_type is not None:
        query_params.update(
            {
                "unit_type": sample.highlighted_unit_type,
                "unit_index": sample.highlighted_unit_index,
                "unit_type_probabilities": dict(axes.unit_type_probabilities or {}),
            }
        )
    prompt_spec_payload = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=query_params,
    )
    trace_payload = build_sudoku_trace_payload(
        sample=sample,
        rendered_scene=visual.rendered_scene,
        image=visual.image,
        prompt_artifacts=prompt_artifacts,
        prompt_spec_payload=prompt_spec_payload,
        execution_fields={
            "query_id": str(selected_query_id),
            "answer_label": str(sample.answer),
            "option_count": len(axes.option_labels),
            "target_answer_support": [str(label) for label in axes.option_labels],
            "target_answer_probabilities": dict(axes.answer_label_probabilities),
        },
        annotation_type=str(annotation.annotation_kind),
        annotation_value=annotation.annotation_value,
        annotation_entity_ids=annotation.entity_ids,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        panel_style_meta=visual.panel_style_meta,
        text_style_meta=text_style_metadata(str(render_params.font_family)),
        background_meta=visual.background_meta,
        post_noise_meta=visual.post_noise_meta,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type="option_letter", value=str(sample.answer)),
        annotation_gt=TypedValue(
            type=str(annotation.annotation_kind),
            value=annotation.annotation_value,
        ),
        image=visual.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


def run_sudoku_single_query_lifecycle(
    *,
    runtime: SudokuTaskRuntime,
    params: Mapping[str, Any],
    build_sample: SampleBuilder,
    instance_seed: int,
    max_attempts: int,
) -> TaskOutput:
    """Select the fixed Sudoku branch, then run neutral scene lifecycle plumbing."""

    defaults = SudokuDefaults()
    gen_defaults, render_defaults, prompt_defaults = (
        load_scene_generation_rendering_prompt_defaults(
            "puzzles",
            SCENE_ID,
            task_id=str(runtime.source_id),
        )
    )
    fallback_support = getattr(defaults, str(runtime.support_key))
    prompt_required_keys = (
        "bundle_id",
        "scene_key",
        "task_key",
        "object_description_sparse_grid",
        "object_description_filled_grid",
    )
    if bool(runtime.include_unit_type):
        prompt_required_keys = (
            *prompt_required_keys,
            "unit_scope_text_row",
            "unit_scope_text_column",
            "unit_scope_text_box",
        )
    noise_defaults = load_puzzle_noise_defaults(
        scene_id=SCENE_ID,
        apply_prob=0.5,
    )
    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=(SINGLE_QUERY_ID,),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(runtime.source_id),
        namespace=f"{runtime.attempt_namespace}.query",
    )
    return run_sudoku_lifecycle(
        source_id=str(runtime.source_id),
        selected_query_id=str(selected_query),
        query_probabilities=query_probabilities,
        params=task_params,
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        defaults=defaults,
        support_key=str(runtime.support_key),
        fallback_support=fallback_support,
        include_unit_type=bool(runtime.include_unit_type),
        prompt_query_key=str(runtime.prompt_query_key),
        prompt_required_keys=prompt_required_keys,
        attempt_namespace=str(runtime.attempt_namespace),
        build_sample=build_sample,
        noise_defaults=noise_defaults,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
    )


def run_sudoku_single_query_option_lifecycle(
    *,
    source_id: str,
    prompt_query_key: str,
    attempt_namespace: str,
    build_sample: SampleBuilder,
    params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    include_unit_type: bool = False,
    target_digit_support_key: str | None = None,
) -> TaskOutput:
    """Select the fixed Sudoku branch, then run option-letter lifecycle plumbing."""

    defaults = SudokuDefaults()
    gen_defaults, render_defaults, prompt_defaults = (
        load_scene_generation_rendering_prompt_defaults(
            "puzzles",
            SCENE_ID,
            task_id=str(source_id),
        )
    )
    prompt_required_keys = (
        "bundle_id",
        "scene_key",
        "task_key",
        "object_description_sparse_grid",
        "object_description_filled_grid",
    )
    if bool(include_unit_type):
        prompt_required_keys = (
            *prompt_required_keys,
            "unit_scope_text_row",
            "unit_scope_text_column",
            "unit_scope_text_box",
        )
    noise_defaults = load_puzzle_noise_defaults(
        scene_id=SCENE_ID,
        apply_prob=0.5,
    )
    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=(SINGLE_QUERY_ID,),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(source_id),
        namespace=f"{attempt_namespace}.query",
    )
    return run_sudoku_option_lifecycle(
        source_id=str(source_id),
        selected_query_id=str(selected_query),
        query_probabilities=query_probabilities,
        params=task_params,
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        defaults=defaults,
        include_unit_type=bool(include_unit_type),
        target_digit_support_key=target_digit_support_key,
        prompt_query_key=str(prompt_query_key),
        prompt_required_keys=prompt_required_keys,
        attempt_namespace=str(attempt_namespace),
        build_sample=build_sample,
        noise_defaults=noise_defaults,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
    )


__all__ = [
    "RenderedSudokuScene",
    "SudokuAnnotationBinding",
    "SudokuOptionAxes",
    "SudokuTaskRuntime",
    "run_sudoku_lifecycle",
    "run_sudoku_option_lifecycle",
    "run_sudoku_single_query_lifecycle",
    "run_sudoku_single_query_option_lifecycle",
]
