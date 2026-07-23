"""Private lifecycle shell for symbolic agent-automaton option tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions

from .shared.annotations import annotation_trace_payload, keyed_bboxes
from .shared.output import agent_run_execution_fields, build_agent_trace_payload
from .shared.prompts import build_agent_prompt
from .shared.rendering import render_agent_scene_bundle
from .shared.rules import SCENE_ID, state_count_for_rule
from .shared.sampling import sample_agent_run
from .shared.state import AgentRenderBundle, AgentSceneSpec, AgentStepTrace
from .shared.styles import resolve_rule_variant, resolve_scene_variant


@dataclass(frozen=True)
class AgentObjectiveResult:
    """Task-owned answer, annotation, and trace extras for a rendered scene."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    answer_value: Any
    witness_symbolic: Mapping[str, Any]
    projected_annotation: Mapping[str, Any]
    execution_fields: Mapping[str, Any]
    render_map_extra: Mapping[str, Any]


@dataclass(frozen=True)
class AgentOptionDataset:
    """Common simulated agent run plus task-specific option payload."""

    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    rule_variant: str
    rule_variant_probabilities: dict[str, float]
    rows: int
    cols: int
    steps: int
    initial_grid: Tuple[Tuple[int, ...], ...]
    final_grid: Tuple[Tuple[int, ...], ...]
    start_row: int
    start_col: int
    start_direction: int
    final_row: int
    final_col: int
    final_direction: int
    traces: Tuple[AgentStepTrace, ...]
    option_specs: Tuple[Any, ...]
    answer_label: str


@dataclass(frozen=True)
class AgentOptionLifecycleBinding:
    """Task-local hooks and prompt keys for one option-label objective."""

    domain: str
    task_identifier: str
    question_format: str
    internal_query_id: str
    supported_query_ids: Sequence[str]
    task_prompt_key: str
    question_text_key: str
    annotation_hint_key: str
    answer_hint_key: str
    json_example_key: str
    json_example_answer_only_key: str
    step_min_key: str
    step_max_key: str
    fallback_step_min: int
    fallback_step_max: int
    option_mode: str
    choose_options: Callable[..., tuple[Tuple[Any, ...], str]]
    build_objective: Callable[[AgentOptionDataset, AgentRenderBundle], AgentObjectiveResult]


def make_agent_option_binding(
    task_identifier: str,
    question_format: str,
    task_prompt_key: str,
    step_min_key: str,
    step_max_key: str,
    option_mode: str,
    choose_options: Callable[..., tuple[Tuple[Any, ...], str]],
    build_objective: Callable[[AgentOptionDataset, AgentRenderBundle], AgentObjectiveResult],
    prompt_key_suffix: str | None = None,
) -> AgentOptionLifecycleBinding:
    """Create the standard single-query binding for an agent option task."""

    suffix = str(prompt_key_suffix) if prompt_key_suffix else ""
    key = lambda base: f"{base}_{suffix}" if suffix else base
    return AgentOptionLifecycleBinding(
        domain="symbolic",
        task_identifier=str(task_identifier),
        question_format=str(question_format),
        internal_query_id=str(question_format),
        supported_query_ids=("single",),
        task_prompt_key=str(task_prompt_key),
        question_text_key=key("question_text"),
        annotation_hint_key=key("annotation_hint"),
        answer_hint_key=key("answer_hint"),
        json_example_key=key("json_example"),
        json_example_answer_only_key=key("json_example_answer_only"),
        step_min_key=str(step_min_key),
        step_max_key=str(step_max_key),
        fallback_step_min=3,
        fallback_step_max=6,
        option_mode=str(option_mode),
        choose_options=choose_options,
        build_objective=build_objective,
    )


def pose_record(dataset: AgentOptionDataset, *, use_direction_names: bool) -> dict[str, Any]:
    """Serialize start/final pose metadata for task traces."""

    def direction(value: int) -> Any:
        from .shared.rules import DIRECTIONS

        return DIRECTIONS[int(value)] if bool(use_direction_names) else int(value)

    return {
        "start_pose": {"row": int(dataset.start_row), "col": int(dataset.start_col), "direction": direction(dataset.start_direction)},
        "final_pose": {"row": int(dataset.final_row), "col": int(dataset.final_col), "direction": direction(dataset.final_direction)},
    }


def pose_option_records(option_specs: Sequence[Any]) -> list[dict[str, Any]]:
    """Serialize pose option cards for replay/debug metadata."""

    return [
        {
            "option_id": str(option.option_id),
            "label": str(option.label),
            "row": int(option.row),
            "col": int(option.col),
            "direction": int(option.direction),
            "pose_text": str(option.pose_text),
            "is_correct": bool(option.is_correct),
        }
        for option in option_specs
    ]


def grid_option_records(option_specs: Sequence[Any]) -> list[dict[str, Any]]:
    """Serialize future-grid option cards for replay/debug metadata."""

    return [
        {
            "option_id": str(option.option_id),
            "label": str(option.label),
            "grid": [list(row) for row in option.grid],
            "is_correct": bool(option.is_correct),
        }
        for option in option_specs
    ]


def step_trace_records(traces: Sequence[AgentStepTrace]) -> list[dict[str, int]]:
    """Serialize simulated update steps for replay/debug metadata."""

    return [
        {
            "step": int(trace.step),
            "row": int(trace.row),
            "col": int(trace.col),
            "direction": int(trace.direction),
            "state_before": int(trace.state_before),
            "state_after": int(trace.state_after),
        }
        for trace in traces
    ]


def build_agent_option_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    rule_variant: str,
    rule_variant_probabilities: Mapping[str, float],
    question_format: str,
    step_min_key: str,
    step_max_key: str,
    fallback_step_min: int,
    fallback_step_max: int,
    choose_options: Callable[..., tuple[Tuple[Any, ...], str]],
) -> AgentOptionDataset:
    """Build one simulated run and attach task-supplied visual options."""

    run = sample_agent_run(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sample_scope=f"{question_format}.agent",
        step_min_key=str(step_min_key),
        step_max_key=str(step_max_key),
        fallback_step_min=int(fallback_step_min),
        fallback_step_max=int(fallback_step_max),
        rule_variant=str(rule_variant),
    )
    option_specs, answer_label = choose_options(
        run=run,
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rule_variant=str(rule_variant),
    )
    return AgentOptionDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        rule_variant=str(rule_variant),
        rule_variant_probabilities={str(key): float(value) for key, value in rule_variant_probabilities.items()},
        rows=int(run.rows),
        cols=int(run.cols),
        steps=int(run.steps),
        initial_grid=tuple(run.initial_grid),
        final_grid=tuple(run.final_grid),
        start_row=int(run.start_row),
        start_col=int(run.start_col),
        start_direction=int(run.start_direction),
        final_row=int(run.final_row),
        final_col=int(run.final_col),
        final_direction=int(run.final_direction),
        traces=tuple(run.traces),
        option_specs=tuple(option_specs),
        answer_label=str(answer_label),
    )


def agent_option_scene_spec(dataset: AgentOptionDataset, *, option_mode: str) -> AgentSceneSpec:
    """Create the passive render spec for the selected option-card family."""

    option_kwargs: dict[str, Any]
    if str(option_mode) == "pose":
        option_kwargs = {"option_specs": tuple(dataset.option_specs)}
    elif str(option_mode) == "grid":
        option_kwargs = {
            "grid_option_specs": tuple(dataset.option_specs),
            "source_marker_label": "START",
        }
    else:
        raise ValueError(f"unsupported agent option mode: {option_mode!r}")
    return AgentSceneSpec(
        rows=int(dataset.rows),
        cols=int(dataset.cols),
        state_count=state_count_for_rule(str(dataset.rule_variant)),
        initial_grid=tuple(dataset.initial_grid),
        start_row=int(dataset.start_row),
        start_col=int(dataset.start_col),
        start_direction=int(dataset.start_direction),
        traces=tuple(dataset.traces),
        **option_kwargs,
    )


def bbox_map_option_result(
    *,
    answer_label: str,
    item_bboxes: Mapping[str, Any],
    annotation_role_item_ids: Mapping[str, str],
    execution_fields: Mapping[str, Any],
    render_map_extra: Mapping[str, Any],
) -> AgentObjectiveResult:
    """Build the common answer/annotation shell for option-card objectives."""

    annotation_value = keyed_bboxes(item_bboxes, annotation_role_item_ids)
    witness_symbolic, projected_annotation = annotation_trace_payload(
        annotation_type="bbox_map",
        annotation_value=annotation_value,
    )
    return AgentObjectiveResult(
        answer_gt=TypedValue(type="option_letter", value=str(answer_label)),
        annotation_gt=TypedValue(type="bbox_map", value=dict(annotation_value)),
        answer_value=str(answer_label),
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
        execution_fields={
            **dict(execution_fields),
            "supporting_item_ids": list(annotation_role_item_ids.values()),
            "supporting_item_ids_by_role": dict(annotation_role_item_ids),
        },
        render_map_extra=dict(render_map_extra),
    )


def run_bound_agent_option_lifecycle(
    binding: AgentOptionLifecycleBinding,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
) -> TaskOutput:
    """Load defaults and run one bound symbolic agent option task."""

    gen_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        binding.domain,
        SCENE_ID,
        task_id=binding.task_identifier,
    )

    def build_dataset(**kwargs: Any) -> AgentOptionDataset:
        return build_agent_option_dataset(
            **kwargs,
            gen_defaults=gen_defaults,
            question_format=binding.question_format,
            step_min_key=binding.step_min_key,
            step_max_key=binding.step_max_key,
            fallback_step_min=binding.fallback_step_min,
            fallback_step_max=binding.fallback_step_max,
            choose_options=binding.choose_options,
        )

    return run_agent_option_lifecycle(
        domain=binding.domain,
        task_id=binding.task_identifier,
        question_format=binding.question_format,
        internal_query_id=binding.internal_query_id,
        supported_query_ids=binding.supported_query_ids,
        task_prompt_key=binding.task_prompt_key,
        question_text_key=binding.question_text_key,
        annotation_hint_key=binding.annotation_hint_key,
        answer_hint_key=binding.answer_hint_key,
        json_example_key=binding.json_example_key,
        json_example_answer_only_key=binding.json_example_answer_only_key,
        params=params,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        build_dataset=build_dataset,
        build_scene=lambda dataset: agent_option_scene_spec(dataset, option_mode=binding.option_mode),
        build_objective=binding.build_objective,
    )


def run_agent_option_lifecycle(
    *,
    domain: str,
    task_id: str,
    question_format: str,
    internal_query_id: str,
    supported_query_ids: Sequence[str],
    task_prompt_key: str,
    question_text_key: str,
    annotation_hint_key: str,
    answer_hint_key: str,
    json_example_key: str,
    json_example_answer_only_key: str,
    params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    build_dataset: Callable[..., Any],
    build_scene: Callable[[Any], AgentSceneSpec],
    build_objective: Callable[[Any, AgentRenderBundle], AgentObjectiveResult],
) -> TaskOutput:
    """Run shared scene materialization around task-owned option objectives.

    This helper owns only neutral scene setup: resolving non-semantic scene/rule
    axes, retrying task-supplied dataset construction, rendering the task-supplied
    scene spec, composing external prompt templates, and packaging the final
    trace shell. Public task files still bind the objective answer, annotation,
    selected witnesses, option specs, and task-specific execution fields.
    """

    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(question_format),
    )
    rule_variant, rule_variant_probabilities = resolve_rule_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(question_format),
    )
    last_error: Exception | None = None
    dataset = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            dataset = build_dataset(
                params=params,
                instance_seed=int(instance_seed) + int(attempt_index),
                scene_variant=str(scene_variant),
                scene_variant_probabilities=scene_variant_probabilities,
                rule_variant=str(rule_variant),
                rule_variant_probabilities=rule_variant_probabilities,
            )
            break
        except Exception as exc:  # pragma: no cover - exercised by smoke/review generation.
            last_error = exc
            dataset = None
    if dataset is None:
        raise RuntimeError(f"could not generate {task_id}: {last_error}") from last_error

    render_bundle = render_agent_scene_bundle(
        scene=build_scene(dataset),
        params=params,
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        scene_variant=str(dataset.scene_variant),
        instance_seed=int(instance_seed),
        sampling_scope=str(question_format),
    )
    prompt, prompt_variants, _prompt_meta, prompt_artifacts = build_agent_prompt(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        scene_variant=str(dataset.scene_variant),
        rule_variant=str(dataset.rule_variant),
        steps=int(dataset.steps),
        instance_seed=int(instance_seed),
        prompt_key=str(task_prompt_key),
        question_text_key=str(question_text_key),
        annotation_hint_key=str(annotation_hint_key),
        answer_hint_key=str(answer_hint_key),
        json_example_key=str(json_example_key),
        json_example_answer_only_key=str(json_example_answer_only_key),
    )
    objective = build_objective(dataset, render_bundle)
    query_params = {
        "query_id": str(supported_query_ids[0]),
        "query_id_probabilities": {str(supported_query_ids[0]): 1.0},
        "internal_query_id": str(internal_query_id),
        "question_format": str(question_format),
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.scene_variant),
        "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
        "rule_variant": str(dataset.rule_variant),
        "rule_variant_probabilities": dict(dataset.rule_variant_probabilities),
        "agent_board_style": str(render_bundle.board_style),
        "agent_board_style_probabilities": dict(render_bundle.board_style_probabilities),
    }
    execution_trace = {
        **dict(query_params),
        "task_id": str(task_id),
        "answer_value": objective.answer_value,
        **agent_run_execution_fields(
            rows=dataset.rows,
            cols=dataset.cols,
            steps=dataset.steps,
            rule_variant=dataset.rule_variant,
            initial_grid=dataset.initial_grid,
            final_grid=dataset.final_grid,
        ),
        **dict(objective.execution_fields),
    }
    trace_payload = build_agent_trace_payload(
        scene_name=SCENE_ID,
        prompt_artifacts=prompt_artifacts,
        branch_name=str(supported_query_ids[0]),
        params_payload=query_params,
        render_bundle=render_bundle,
        execution_record=execution_trace,
        witness_symbolic=objective.witness_symbolic,
        projected_annotation=objective.projected_annotation,
        render_map_extra=dict(objective.render_map_extra),
    )
    return TaskOutput(
        prompt=prompt,
        answer_gt=objective.answer_gt,
        annotation_gt=objective.annotation_gt,
        image=render_bundle.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(supported_query_ids[0]),
        prompt_variants=prompt_variants,
    )
