"""Private scene lifecycle helpers for symbolic Turing tape tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ..shared.scene_style import make_symbolic_scene_background

from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, load_turing_defaults, resolve_render_params, resolve_scene_variant
from .shared.output import build_turing_trace_payload
from .shared.prompts import render_turing_prompt
from .shared.rendering import render_turing_scene
from .shared.rules import build_turing_dataset
from .shared.state import SCENE_ID, RenderedTuringScene, TuringDataset, TuringRenderParams
from .shared.styles import resolve_turing_style


@dataclass(frozen=True)
class PreparedTuringScene:
    """Resolved scene assets shared by Turing tape task objectives."""

    gen_defaults: Mapping[str, Any]
    render_defaults: Mapping[str, Any]
    prompt_defaults: Mapping[str, Any]
    public_query_id: str
    query_probabilities: Dict[str, float]
    task_params: Dict[str, Any]
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    render_params: TuringRenderParams
    dataset: TuringDataset
    rendered: RenderedTuringScene
    image: Any
    post_noise_meta: Dict[str, Any]
    prompt: str
    prompt_variants: Dict[str, str]
    prompt_artifacts: Any
    background_meta: Dict[str, Any]


@dataclass(frozen=True)
class TuringObjectiveResult:
    """Task-owned objective binding for one prepared Turing scene."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    answer_value: Any
    witness_symbolic: Mapping[str, Any]
    projected_annotation: Mapping[str, Any]
    execution_fields: Mapping[str, Any]


def prepare_turing_scene(
    *,
    task_identifier: str,
    supported_query_ids: Sequence[str],
    task_prompt_key: str,
    prompt_query_key: str,
    params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    sampling_namespace: str,
    desired_answer_namespace: str,
    query_symbol_namespace: str,
) -> PreparedTuringScene:
    """Prepare the common tape-machine dataset, rendering, prompt, and image.

    Public task files still bind objective-specific answers, annotations, trace
    fields, and final TaskOutput objects.
    """

    gen_defaults, render_defaults, prompt_defaults = load_turing_defaults(str(task_identifier))
    public_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        task_id=str(task_identifier),
        namespace=f"{task_identifier}.query",
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        params=task_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        task_identifier=str(task_identifier),
    )
    render_params = resolve_render_params(task_params, render_defaults, instance_seed=int(instance_seed))
    style, style_meta = resolve_turing_style(scene_variant=str(scene_variant), render_params=render_params)
    background, background_meta = make_symbolic_scene_background(
        canvas_width=render_params.canvas_width,
        canvas_height=render_params.canvas_height,
        style=style,
    )

    last_error: Exception | None = None
    dataset = None
    rendered = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            dataset = build_turing_dataset(
                params=task_params,
                gen_defaults=gen_defaults,
                instance_seed=int(instance_seed) + int(attempt_index),
                sampling_namespace=str(sampling_namespace),
                desired_answer_namespace=str(desired_answer_namespace),
                query_symbol_namespace=str(query_symbol_namespace),
            )
            rendered = render_turing_scene(
                background=background,
                dataset=dataset,
                scene_variant=str(scene_variant),
                render_params=render_params,
                style=style,
                style_meta=style_meta,
            )
            break
        except Exception as exc:  # pragma: no cover - retry guard for constrained samples
            last_error = exc
    if dataset is None or rendered is None:
        raise RuntimeError("failed to construct a valid Turing tape instance") from last_error

    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt, prompt_variants, prompt_artifacts = render_turing_prompt(
        prompt_defaults=prompt_defaults,
        scene_variant=str(scene_variant),
        task_prompt_key=str(task_prompt_key),
        prompt_query_key=str(prompt_query_key),
        steps=int(dataset.steps),
        query_symbol=str(dataset.query_symbol),
        instance_seed=int(instance_seed),
    )
    return PreparedTuringScene(
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        public_query_id=str(public_query_id),
        query_probabilities=dict(query_probabilities),
        task_params=dict(task_params),
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        render_params=render_params,
        dataset=dataset,
        rendered=rendered,
        image=image,
        post_noise_meta=dict(post_noise_meta),
        prompt=str(prompt),
        prompt_variants=dict(prompt_variants),
        prompt_artifacts=prompt_artifacts,
        background_meta=dict(background_meta),
    )


def turing_machine_execution_fields(dataset: TuringDataset) -> Dict[str, Any]:
    """Serialize the objective-neutral Turing machine execution record."""

    return {
        "steps": int(dataset.steps),
        "tape_length": int(dataset.tape_length),
        "symbol_count": int(dataset.symbol_count),
        "symbols": list(dataset.symbols),
        "query_symbol": str(dataset.query_symbol),
        "start_state": str(dataset.start_state),
        "start_head": int(dataset.start_head),
        "initial_tape": list(dataset.initial_tape),
        "final_tape": list(dataset.final_tape),
        "transitions": [
            {
                "state": transition.state,
                "read_symbol": transition.read_symbol,
                "write_symbol": transition.write_symbol,
                "move": transition.move,
                "next_state": transition.next_state,
            }
            for transition in dataset.transitions
        ],
        "step_trace": [
            {
                "step": trace.step,
                "state": trace.state,
                "head_position": trace.head_position,
                "read_symbol": trace.read_symbol,
                "write_symbol": trace.write_symbol,
                "move": trace.move,
                "next_state": trace.next_state,
            }
            for trace in dataset.traces
        ],
    }


def run_turing_lifecycle(
    *,
    task_identifier: str,
    supported_query_ids: Sequence[str],
    task_prompt_key: str,
    prompt_query_key: str,
    params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    build_objective: Callable[[PreparedTuringScene], TuringObjectiveResult],
) -> TaskOutput:
    """Run common Turing scene materialization around a task-owned objective."""

    prepared = prepare_turing_scene(
        task_identifier=str(task_identifier),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        task_prompt_key=str(task_prompt_key),
        prompt_query_key=str(prompt_query_key),
        params=params,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        sampling_namespace=f"{task_identifier}.dataset",
        desired_answer_namespace=f"{task_identifier}.desired_answer_count",
        query_symbol_namespace=f"{task_identifier}.query_symbol",
    )
    objective = build_objective(prepared)
    execution_trace = {
        **turing_machine_execution_fields(prepared.dataset),
        **dict(objective.execution_fields),
    }
    trace_payload = build_turing_trace_payload(
        scene_name=SCENE_ID,
        prompt_artifacts=prepared.prompt_artifacts,
        public_query_id=str(prepared.public_query_id),
        params_payload={
            "query_id": str(prepared.public_query_id),
            "prompt_query_key": str(prompt_query_key),
            "question_format": str(prompt_query_key),
            "scene_id": SCENE_ID,
            "scene_variant": str(prepared.scene_variant),
            "scene_variant_probabilities": dict(prepared.scene_variant_probabilities),
            "query_id_probabilities": dict(prepared.query_probabilities),
        },
        render_params=prepared.render_params,
        rendered_scene=prepared.rendered,
        background_meta=prepared.background_meta,
        post_noise_meta=prepared.post_noise_meta,
        answer_value=objective.answer_value,
        execution_record=execution_trace,
        witness_symbolic=objective.witness_symbolic,
        projected_annotation=objective.projected_annotation,
    )
    return TaskOutput(
        prompt=str(prepared.prompt),
        answer_gt=objective.answer_gt,
        annotation_gt=objective.annotation_gt,
        image=prepared.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(prepared.public_query_id),
        prompt_variants=prepared.prompt_variants,
    )


__all__ = [
    "PreparedTuringScene",
    "TuringObjectiveResult",
    "prepare_turing_scene",
    "run_turing_lifecycle",
    "turing_machine_execution_fields",
]
