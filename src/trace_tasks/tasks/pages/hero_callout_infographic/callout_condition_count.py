"""Hero-callout task for counting callouts satisfying one value condition."""

from __future__ import annotations

from typing import Any, Dict, List

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from . import _lifecycle


TASK_ID = "task_pages__hero_callout_infographic__callout_condition_count"
ABOVE_THRESHOLD_QUERY_ID = "field_value_above_threshold_count"
BELOW_THRESHOLD_QUERY_ID = "field_value_below_threshold_count"
SUPPORTED_QUERY_IDS = (ABOVE_THRESHOLD_QUERY_ID, BELOW_THRESHOLD_QUERY_ID)
PROMPT_QUERY_KEY = "callout_condition_count"
QUESTION_FORMAT = "hero_callout_infographic_callout_condition_count"


def _operator_for_query_id(query_id: str) -> str:
    """Map the public semantic query branch to its threshold operator."""

    if str(query_id) == ABOVE_THRESHOLD_QUERY_ID:
        return "above"
    if str(query_id) == BELOW_THRESHOLD_QUERY_ID:
        return "below"
    raise ValueError(f"unsupported callout condition query_id: {query_id}")


def _build_prompt_slots(target: Dict[str, Any]) -> Dict[str, str]:
    """Bind the resolved field, threshold, and predicate into prompt slots."""

    return {
        "field_label": f'"{target["field_label"]}"',
        "condition_phrase": str(target["condition_phrase"]),
        "threshold_value": f'"{target["threshold_visible"]}"',
    }


def _condition_matching_boxes(ctx: Any, target: Dict[str, Any]) -> List[List[float]]:
    """Project the field rows that satisfy the selected threshold predicate."""

    matching_boxes: List[List[float]] = []
    for match in target["matching_values"]:
        match_callout = str(match["callout_id"])
        match_field = str(match["field_id"])
        matching_boxes.append(
            [float(value) for value in ctx.rendered.field_row_bboxes_px[match_callout][match_field]]
        )
    return matching_boxes


def _condition_candidate_partition(target: Dict[str, Any]) -> Dict[str, List[str] | int]:
    """Record which evaluated callouts matched and which remained outside the count."""

    matched_ids = {str(match["callout_id"]) for match in target["matching_values"]}
    outside_ids: List[str] = []
    inside_ids: List[str] = []
    for candidate in target["candidate_values"]:
        candidate_id = str(candidate["callout_id"])
        if candidate_id in matched_ids:
            inside_ids.append(candidate_id)
        else:
            outside_ids.append(candidate_id)
    return {
        "matching_callout_ids": inside_ids,
        "nonmatching_callout_ids": outside_ids,
        "evaluated_candidate_count": int(len(inside_ids) + len(outside_ids)),
    }


def _condition_trace_extras(
    field_probs: Dict[str, float],
    threshold_probs: Dict[str, float],
    target: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Prepare predicate sampling metadata and matching-callout annotation keys."""

    query_params_extra = {
        "target_field_label_probabilities": dict(field_probs),
        "threshold_rank_index_probabilities": dict(threshold_probs),
    }
    execution_extra = {
        "condition_operator": str(target["condition_operator"]),
        "threshold_value": int(target["threshold_value"]),
        "threshold_visible": str(target["threshold_visible"]),
        "condition_partition": _condition_candidate_partition(target),
    }
    annotation_keys = [str(match["callout_id"]) for match in target["matching_values"]]
    return query_params_extra, execution_extra, annotation_keys


@register_task
class PagesHeroCalloutConditionCountTask:
    """Count callouts whose value satisfies one visible threshold condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one threshold-count query and bind matching field-row boxes."""

        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=ABOVE_THRESHOLD_QUERY_ID,
            public_task=TASK_ID,
        )
        task_params = dict(task_params)
        task_params["condition_operator"] = _operator_for_query_id(str(selected_branch))
        ctx = _lifecycle.resolve_scene_context(
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        target, field_probs, threshold_probs = _lifecycle.select_condition_target(
            gen_defaults=ctx.gen_defaults,
            spec=ctx.spec,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation = _condition_matching_boxes(ctx, target)
        query_params_extra, execution_extra, annotation_keys = _condition_trace_extras(
            field_probs,
            threshold_probs,
            target,
        )
        prompt_artifacts = _lifecycle.render_prompt(
            prompt_query_key=PROMPT_QUERY_KEY,
            dynamic_slots=_build_prompt_slots(target),
            instance_seed=int(instance_seed),
        )
        trace_payload = _lifecycle._trace_payload(
            ctx=ctx,
            prompt_artifacts=prompt_artifacts,
            prompt_query_key=PROMPT_QUERY_KEY,
            question_format=QUESTION_FORMAT,
            target_payload=target,
            answer_value=int(target["answer_value"]),
            annotation_type="bbox_set",
            annotation_value=list(annotation),
            annotation_keys=annotation_keys,
            query_params_extra=query_params_extra,
            execution_extra=execution_extra,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(target["answer_value"])),
            annotation_gt=TypedValue(type="bbox_set", value=list(annotation)),
            image=ctx.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=_lifecycle.SCENE_ID,
            query_id=str(ctx.selected_branch),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = [
    "ABOVE_THRESHOLD_QUERY_ID",
    "BELOW_THRESHOLD_QUERY_ID",
    "PROMPT_QUERY_KEY",
    "QUESTION_FORMAT",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesHeroCalloutConditionCountTask",
]
