"""Public task for `task_charts__waterfall__signed_step_threshold_count`."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.shared.label_assets import sample_chart_labels
from trace_tasks.tasks.charts.waterfall._lifecycle import (
    WaterfallTaskPlan,
    run_waterfall_lifecycle,
)
from trace_tasks.tasks.charts.waterfall.shared.annotations import bbox_set_artifacts
from trace_tasks.tasks.charts.waterfall.shared.defaults import (
    DOMAIN,
    GENERATION_DEFAULTS,
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.charts.waterfall.shared.state import WaterfallDataset, WaterfallStep
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice


TASK_ID = "task_charts__waterfall__signed_step_threshold_count"
OBJECTIVE_CONTRACT = "signed_step_threshold_count"
PROMPT_QUERY_KEY = "signed_step_threshold_count"
DECREASE_QUERY_ID = "decrease_step_threshold_count"
INCREASE_QUERY_ID = "increase_step_threshold_count"
SUPPORTED_QUERY_IDS = (DECREASE_QUERY_ID, INCREASE_QUERY_ID)
DEFAULT_QUERY_ID = DECREASE_QUERY_ID
DIRECTION_WORDS = {
    DECREASE_QUERY_ID: ("decreases", "drop"),
    INCREASE_QUERY_ID: ("increases", "rise"),
}
SIGN_BY_QUERY_ID = {DECREASE_QUERY_ID: -1, INCREASE_QUERY_ID: 1}
# Sampled before the step count so every answer stays equally likely; the step
# count is then widened to fit the drawn target.
SIGNED_STEP_THRESHOLD_COUNT_SUPPORT = (1, 2, 3, 4, 5)
THRESHOLD_SUPPORT = (8, 10, 12)


def _support_sampling_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Decouple the sampling cursor from the public query-id round robin.

    Finite-support callers cycle `_sample_cursor` while the query id rotates in
    lockstep, so an undivided cursor would only ever reach the support entries
    whose index shares the query's residue.
    """

    support_params = dict(params)
    cursor = support_params.get("_sample_cursor")
    if cursor is None:
        return support_params
    support_params["_sample_cursor"] = abs(int(cursor)) // max(1, len(SUPPORTED_QUERY_IDS))
    return support_params


def _resolve_target_count(params: Mapping[str, Any], *, instance_seed: int) -> int:
    """Resolve how many same-signed steps must clear the magnitude threshold."""

    target, _probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=_support_sampling_params(params),
        gen_defaults=GENERATION_DEFAULTS,
        support_key="signed_step_threshold_count_support",
        explicit_key="target_answer",
        fallback_support=SIGNED_STEP_THRESHOLD_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return int(target)


def _resolve_threshold(params: Mapping[str, Any], *, instance_seed: int) -> int:
    """Resolve the prompt-bound magnitude threshold."""

    requested = params.get("magnitude_threshold")
    support = tuple(int(v) for v in params.get("magnitude_threshold_support", THRESHOLD_SUPPORT))
    if requested is not None:
        if int(requested) not in support:
            raise ValueError(f"unsupported magnitude_threshold: {requested}")
        return int(requested)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.magnitude_threshold")
    return int(support[rng.randrange(len(support))])


def _build_dataset(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    sign: int,
    matching_count: int,
    decoy_count: int,
    opposite_count: int,
    threshold: int,
) -> WaterfallDataset:
    """Construct a sequence with an exact above-threshold same-signed count.

    Every instance also carries at least one same-signed step *below* the
    threshold, so the colour of the contribution bars alone never answers the
    question.
    """

    start_min, start_max = resolve_required_int_bounds(
        params, GENERATION_DEFAULTS, min_key="start_value_min", max_key="start_value_max",
        fallback_min=35, fallback_max=65, context="waterfall start value",
    )
    delta_min, delta_max = resolve_required_int_bounds(
        params, GENERATION_DEFAULTS, min_key="delta_abs_min", max_key="delta_abs_max",
        fallback_min=5, fallback_max=20, context="waterfall delta magnitude",
    )
    total_min, total_max = resolve_required_int_bounds(
        params, GENERATION_DEFAULTS, min_key="running_total_min", max_key="running_total_max",
        fallback_min=8, fallback_max=95, context="waterfall running total",
    )
    if int(threshold) >= int(delta_max):
        raise ValueError("threshold leaves no above-threshold magnitude room")
    if int(threshold) < int(delta_min):
        raise ValueError("threshold leaves no below-threshold magnitude room")

    classes = (
        ["match"] * int(matching_count)
        + ["decoy"] * int(decoy_count)
        + ["opposite"] * int(opposite_count)
    )
    for attempt in range(200):
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.signed_step_threshold.retry", int(attempt))
        order = list(classes)
        rng.shuffle(order)
        start_value = int(rng.randint(int(start_min), int(start_max)))
        labels = sample_chart_labels(
            count=len(order),
            instance_seed=int(hash64(int(instance_seed), "charts.waterfall.signed_step_threshold.labels", int(attempt))),
            namespace=f"{SCENE_NAMESPACE}.labels:{len(order)}",
        )
        current = int(start_value)
        steps: list[WaterfallStep] = []
        for index, kind in enumerate(order):
            if kind == "match":
                step_sign, low, high = int(sign), int(threshold) + 1, int(delta_max)
            elif kind == "decoy":
                step_sign, low, high = int(sign), int(delta_min), int(threshold)
            else:
                step_sign, low, high = -int(sign), int(delta_min), int(delta_max)
            # Keep the running total inside the scene's configured window.
            headroom = int(total_max) - int(current) if step_sign > 0 else int(current) - int(total_min)
            high = min(int(high), int(headroom))
            if int(high) < int(low):
                break
            magnitude = int(rng.randint(int(low), int(high)))
            delta = int(step_sign) * int(magnitude)
            steps.append(
                WaterfallStep(
                    step_id=f"step_{index}",
                    label=str(labels[index]),
                    delta=int(delta),
                    running_before=int(current),
                    running_after=int(current) + int(delta),
                )
            )
            current = int(current) + int(delta)
        if len(steps) != len(order):
            continue
        matching = [s for s in steps if (1 if int(s.delta) > 0 else -1) == int(sign) and abs(int(s.delta)) > int(threshold)]
        same_sign = [s for s in steps if (1 if int(s.delta) > 0 else -1) == int(sign)]
        if len(matching) != int(matching_count):
            raise ValueError("waterfall threshold construction lost its magnitude plan")
        if len(same_sign) <= len(matching):
            raise ValueError("waterfall threshold instance has no same-signed decoy step")
        return WaterfallDataset(
            start_value=int(start_value),
            final_value=int(steps[-1].running_after),
            steps=tuple(steps),
        )
    raise ValueError("could not construct a waterfall sequence for the requested threshold plan")


def _build_plan(params, instance_seed, selected_branch, query_probabilities):
    """Bind the signed magnitude-threshold count objective before rendering."""

    if str(selected_branch) not in SUPPORTED_QUERY_IDS:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    step_min, step_max = resolve_required_int_bounds(
        params, GENERATION_DEFAULTS, min_key="step_count_min", max_key="step_count_max",
        fallback_min=6, fallback_max=9, context="waterfall step count",
    )
    target_count = _resolve_target_count(params, instance_seed=int(instance_seed))
    threshold = _resolve_threshold(params, instance_seed=int(instance_seed))
    sign = SIGN_BY_QUERY_ID[str(selected_branch)]

    # One same-signed decoy and one opposite-signed step are the minimum that
    # keep the colour cue insufficient and both signs visible.
    lowest_total = max(int(step_min), int(target_count) + 2)
    if lowest_total > int(step_max):
        raise ValueError(f"target_answer {target_count} exceeds the waterfall step-count bounds")
    layout_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.step_layout")
    step_count = int(lowest_total) + int(layout_rng.randrange(int(step_max) - int(lowest_total) + 1))
    remaining = int(step_count) - int(target_count)
    decoy_count = 1 + int(layout_rng.randrange(int(remaining) - 1))
    opposite_count = int(remaining) - int(decoy_count)

    dataset = _build_dataset(
        params,
        instance_seed=int(instance_seed),
        sign=int(sign),
        matching_count=int(target_count),
        decoy_count=int(decoy_count),
        opposite_count=int(opposite_count),
        threshold=int(threshold),
    )
    matching_steps = tuple(
        step for step in dataset.steps
        if (1 if int(step.delta) > 0 else -1) == int(sign) and abs(int(step.delta)) > int(threshold)
    )
    direction_word, motion_word = DIRECTION_WORDS[str(selected_branch)]
    matching_trace = [
        {"step_id": str(s.step_id), "label": str(s.label), "delta": int(s.delta)} for s in matching_steps
    ]
    same_sign_total = sum(
        1 for s in dataset.steps if (1 if int(s.delta) > 0 else -1) == int(sign)
    )

    def _bind_annotation(rendered):
        return bbox_set_artifacts([rendered.bar_bboxes_px[str(s.step_id)] for s in matching_steps])

    return WaterfallTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(target_count)),
        annotation_builder=_bind_annotation,
        prompt_query_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "direction_word": str(direction_word),
            "motion_word": str(motion_word),
            "magnitude_threshold": int(threshold),
        },
        query_params={
            "direction": str(selected_branch),
            "direction_word": str(direction_word),
            "motion_word": str(motion_word),
            "magnitude_threshold": int(threshold),
            "target_answer": int(target_count),
            "matching_step_ids": [str(s.step_id) for s in matching_steps],
            "matching_steps": matching_trace,
            "same_signed_step_count": int(same_sign_total),
            "same_signed_decoy_count": int(same_sign_total) - int(target_count),
            "step_count": int(len(dataset.steps)),
            "start_value": int(dataset.start_value),
            "final_value": int(dataset.final_value),
            "annotation_role": "matching_contribution_bars",
        },
        relations={
            "direction": str(selected_branch),
            "direction_word": str(direction_word),
            "magnitude_threshold": int(threshold),
            "matching_step_ids": [str(s.step_id) for s in matching_steps],
            "same_signed_step_count": int(same_sign_total),
            "query_id_probabilities": dict(query_probabilities),
        },
        witness_symbolic={
            "type": "waterfall_signed_step_threshold_witness",
            "direction": str(selected_branch),
            "magnitude_threshold": int(threshold),
            "matching_step_ids": [str(s.step_id) for s in matching_steps],
            "matching_steps": matching_trace,
            "answer": int(target_count),
        },
        question_format="waterfall_signed_step_threshold_count",
    )


class ChartsWaterfallSignedStepThresholdCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_waterfall_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_plan,
        )


register_task(ChartsWaterfallSignedStepThresholdCountTask)


__all__ = [
    "ChartsWaterfallSignedStepThresholdCountTask",
    "DECREASE_QUERY_ID",
    "INCREASE_QUERY_ID",
    "SIGNED_STEP_THRESHOLD_COUNT_SUPPORT",
]
