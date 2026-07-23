"""Scene-private lifecycle plumbing for cyclic-order puzzle tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
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
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import option_bbox_annotation
from .shared.output import build_trace_payload
from .shared.prompts import build_cyclic_order_prompt_artifacts
from .shared.sampling import (
    resolve_loop_path_style,
    resolve_render_params,
    resolve_scene_variant,
    resolve_token_render_style,
)
from .shared.state import (
    DEFAULTS,
    DOMAIN,
    SCENE_ID,
    TOKEN_STYLE_PROMPT_INSTRUCTION,
)


DatasetBuilder = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class CyclicOrderTaskObjective:
    """Task-owned hooks needed by the neutral cyclic-order lifecycle."""

    prompt_task_key: str
    prompt_query_key: str
    namespace_base: str
    generation_defaults: Mapping[str, Any]
    rendering_defaults: Mapping[str, Any]
    dataset_builder: DatasetBuilder
    scene_renderer: Callable[..., Any]
    render_field_map: Mapping[str, str]
    trace_field_keys: Sequence[str]
    description_by_variant: Mapping[str, str]
    option_sequence_key: str


def run_cyclic_order_objective(
    *,
    task_identity: str,
    objective: CyclicOrderTaskObjective,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run one task-owned cyclic-order objective through the shared lifecycle."""

    return run_cyclic_order_lifecycle(
        task_identity=str(task_identity),
        prompt_task_key=str(objective.prompt_task_key),
        prompt_query_key=str(objective.prompt_query_key),
        namespace_base=str(objective.namespace_base),
        generation_defaults=objective.generation_defaults,
        rendering_defaults=objective.rendering_defaults,
        dataset_builder=objective.dataset_builder,
        scene_renderer=objective.scene_renderer,
        render_field_map=objective.render_field_map,
        trace_field_keys=objective.trace_field_keys,
        description_by_variant=objective.description_by_variant,
        option_sequence_key=str(objective.option_sequence_key),
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
    )


def run_cyclic_order_lifecycle(
    *,
    task_identity: str,
    prompt_task_key: str,
    prompt_query_key: str,
    namespace_base: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    dataset_builder: DatasetBuilder,
    scene_renderer: Callable[..., Any],
    render_field_map: Mapping[str, str],
    trace_field_keys: Sequence[str],
    description_by_variant: Mapping[str, str],
    option_sequence_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run shared cyclic-order scene plumbing with task-owned objective hooks."""

    last_error: Exception | None = None
    attempts = max(1, int(max_attempts))
    for attempt_index in range(attempts):
        try:
            return _generate_once(
                task_identity=str(task_identity),
                prompt_task_key=str(prompt_task_key),
                prompt_query_key=str(prompt_query_key),
                namespace_base=str(namespace_base),
                generation_defaults=generation_defaults,
                rendering_defaults=rendering_defaults,
                dataset_builder=dataset_builder,
                scene_renderer=scene_renderer,
                render_field_map=render_field_map,
                trace_field_keys=trace_field_keys,
                description_by_variant=description_by_variant,
                option_sequence_key=str(option_sequence_key),
                instance_seed=int(instance_seed) + int(attempt_index),
                params=params,
            )
        except ValueError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{task_identity} generation failed without a captured error")


def _generate_once(
    *,
    task_identity: str,
    prompt_task_key: str,
    prompt_query_key: str,
    namespace_base: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    dataset_builder: DatasetBuilder,
    scene_renderer: Callable[..., Any],
    render_field_map: Mapping[str, str],
    trace_field_keys: Sequence[str],
    description_by_variant: Mapping[str, str],
    option_sequence_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> TaskOutput:
    """Generate one cyclic-order output after public files bind semantics."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=(SINGLE_QUERY_ID,),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(task_identity),
        namespace=f"{namespace_base}.branch",
    )
    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported cyclic-order branch: {selected_branch}")

    token_render_style, token_render_style_probabilities = resolve_token_render_style(
        params=task_params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        namespace_base=str(namespace_base),
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        params=task_params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        namespace_base=str(namespace_base),
    )
    loop_path_style, loop_path_style_probabilities = resolve_loop_path_style(
        params=task_params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        namespace_base=str(namespace_base),
    )
    dataset = dataset_builder(
        token_render_style=str(token_render_style),
        loop_path_style=str(loop_path_style),
        params=task_params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        defaults=DEFAULTS,
        namespace_base=str(namespace_base),
    )
    _validate_option_sequences(dataset, option_sequence_key=str(option_sequence_key))

    render_params = resolve_render_params(
        task_params,
        rendering_defaults=rendering_defaults,
        instance_seed=int(instance_seed),
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace_base}.background",
    )
    render_params = replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        instruction_fill_rgb=tuple(int(value) for value in scene_style.option_fill_rgb),
        border_color_rgb=tuple(int(value) for value in scene_style.panel_border_rgb),
        loop_color_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = scene_renderer(
        background,
        scene_variant=str(scene_variant),
        render_params=render_params,
        **_render_kwargs_from_dataset(dataset, render_field_map),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15),
    )

    answer_value = str(dataset["answer_option_label"])
    answer_gt = TypedValue(type="option_letter", value=answer_value)
    annotation_artifacts = option_bbox_annotation(
        rendered_scene.option_choice_bbox_map,
        str(dataset["answer_option_choice_id"]),
    )

    prompt_defaults, prompt_artifacts = build_cyclic_order_prompt_artifacts(
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots={
            "object_description": _description_for_variant(
                str(scene_variant),
                description_by_variant,
            ),
            "token_render_style_instruction": TOKEN_STYLE_PROMPT_INSTRUCTION[str(token_render_style)],
        },
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params={
            "query_id_probabilities": dict(branch_probabilities),
            "token_render_style": str(token_render_style),
            "token_render_style_probabilities": dict(token_render_style_probabilities),
            "loop_path_style": str(loop_path_style),
            "loop_path_style_probabilities": dict(loop_path_style_probabilities),
            "scene_variant": str(scene_variant),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            "answer_option_label_probabilities": dict(dataset["answer_option_label_probabilities"]),
            "option_count": int(dataset["option_count"]),
            "bead_count": int(dataset["bead_count"]),
            "min_color_distance": float(dataset["min_color_distance"]),
            "color_distance_space": str(dataset["color_distance_space"]),
        },
    )

    common_execution = {
        "query_id": str(selected_branch),
        "internal_query_id": str(prompt_query_key),
        "token_render_style": str(token_render_style),
        "bead_token_mode": str(dataset["bead_token_mode"]),
        "loop_path_style": str(loop_path_style),
        "scene_variant": str(scene_variant),
        "query_id_probabilities": dict(branch_probabilities),
        "token_render_style_probabilities": dict(token_render_style_probabilities),
        "loop_path_style_probabilities": dict(loop_path_style_probabilities),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "answer_option_label_probabilities": dict(dataset["answer_option_label_probabilities"]),
        "question_format": str(dataset["question_format"]),
        "view_family": str(dataset["view_family"]),
        "equivalence_rule": str(dataset["equivalence_rule"]),
        "option_specs": [dict(option) for option in dataset["option_specs"]],
        "option_count": int(dataset["option_count"]),
        "option_count_range": list(dataset["option_count_range"]),
        "valid_option_count": int(dataset["valid_option_count"]),
        "bead_count": int(dataset["bead_count"]),
        "bead_count_range": list(dataset["bead_count_range"]),
        "valid_option_choice_ids": list(dataset["valid_option_choice_ids"]),
        "valid_option_labels": [str(value) for value in dataset["valid_option_labels"]],
        "answer_option_choice_id": str(dataset["answer_option_choice_id"]),
        "answer_option_label": str(dataset["answer_option_label"]),
        "supporting_option_choice_ids": list(dataset["valid_option_choice_ids"]),
        "min_color_distance": float(dataset["min_color_distance"]),
        "color_distance_space": str(dataset["color_distance_space"]),
        "answer_value": str(answer_value),
        "solver_trace": dict(dataset["solver_trace"]),
    }
    common_execution.update(_trace_fields_from_dataset(dataset, trace_field_keys))
    trace_payload = build_trace_payload(
        scene_ir={
            "scene_kind": f"puzzle_cyclic_order_{str(dataset['question_format'])}_{str(scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "selected_branch": str(selected_branch),
                "token_render_style": str(token_render_style),
                "bead_token_mode": str(dataset["bead_token_mode"]),
                "loop_path_style": str(loop_path_style),
                "scene_variant": str(scene_variant),
                "answer_value": str(answer_value),
                "equivalence_rule": str(dataset["equivalence_rule"]),
                "valid_option_choice_ids": list(dataset["valid_option_choice_ids"]),
            },
        },
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
        semantic_spec=query_spec,
        render_spec={
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(scene_variant),
            "loop_path_style": str(loop_path_style),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        },
        render_map={
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "reference_loop_bbox_px": list(rendered_scene.reference_loop_bbox_px),
            "option_choice_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.option_choice_bbox_map.items()
            },
        },
        execution_trace=common_execution,
        witness_symbolic={
            "type": "bbox",
            "value": list(annotation_artifacts.value),
        },
        projected_annotation=dict(annotation_artifacts.projected_annotation),
        answer_gt=answer_gt.to_dict(),
        annotation_gt=annotation_artifacts.annotation_gt.to_dict(),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def _render_kwargs_from_dataset(
    dataset: Mapping[str, Any],
    render_field_map: Mapping[str, str],
) -> Dict[str, Any]:
    """Convert task-owned render field maps into renderer keyword arguments."""

    kwargs: Dict[str, Any] = {}
    for output_key, source_key in render_field_map.items():
        value = dataset[str(source_key)]
        if str(output_key).endswith("_specs") or str(output_key) == "option_specs":
            kwargs[str(output_key)] = list(value)
        elif str(output_key).endswith("_deg"):
            kwargs[str(output_key)] = int(value)
        elif str(output_key).endswith("_variant") or str(output_key).endswith("_style"):
            kwargs[str(output_key)] = str(value)
        else:
            kwargs[str(output_key)] = value
    return kwargs


def _trace_fields_from_dataset(
    dataset: Mapping[str, Any],
    trace_field_keys: Sequence[str],
) -> Dict[str, Any]:
    """Copy task-owned trace fields with JSON-friendly scalar/list coercion."""

    fields: Dict[str, Any] = {}
    for key in trace_field_keys:
        value = dataset[str(key)]
        if isinstance(value, (list, tuple)):
            fields[str(key)] = [str(item) for item in value]
        elif isinstance(value, int):
            fields[str(key)] = int(value)
        else:
            fields[str(key)] = str(value)
    return fields


def _description_for_variant(
    scene_variant: str,
    description_by_variant: Mapping[str, str],
) -> str:
    """Resolve prompt object wording from task-owned scene descriptions."""

    return str(
        description_by_variant.get(
            str(scene_variant),
            description_by_variant.get(
                "default",
                "a cyclic-order puzzle diagram",
            ),
        )
    )


def _validate_option_sequences(dataset: Dict[str, Any], *, option_sequence_key: str) -> None:
    """Verify option validity against the task-owned option sequence field."""

    from .shared.rules import token_sequences_are_rotation_equivalent

    reference_tokens = [str(value) for value in dataset["reference_token_sequence"]]
    valid_labels: list[str] = []
    for option in dataset["option_specs"]:
        sequence = [str(value) for value in option[str(option_sequence_key)]]
        is_equivalent = token_sequences_are_rotation_equivalent(reference_tokens, sequence)
        if bool(is_equivalent) != bool(option["is_valid"]):
            raise ValueError("cyclic-order option validity drifted from the task rule")
        if bool(option["is_valid"]):
            valid_labels.append(str(option["option_label"]))
    if valid_labels != [str(dataset["answer_option_label"])]:
        raise ValueError("cyclic-order task must have exactly one valid answer")


__all__ = [
    "CyclicOrderTaskObjective",
    "run_cyclic_order_lifecycle",
    "run_cyclic_order_objective",
]
