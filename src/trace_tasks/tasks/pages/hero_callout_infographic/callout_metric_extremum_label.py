"""Hero-callout task for selecting the extreme callout by metric value."""

from __future__ import annotations

from typing import Any, Dict, List

from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__hero_callout_infographic__callout_metric_extremum_label"
HIGHEST_FIELD_VALUE_QUERY_ID = "highest_field_value_callout_label"
LOWEST_FIELD_VALUE_QUERY_ID = "lowest_field_value_callout_label"
SUPPORTED_QUERY_IDS = (HIGHEST_FIELD_VALUE_QUERY_ID, LOWEST_FIELD_VALUE_QUERY_ID)
PROMPT_QUERY_KEY = "callout_metric_extremum_label"
QUESTION_FORMAT = "hero_callout_infographic_callout_metric_extremum_label"


def _rank_direction_for_query_id(query_id: str) -> str:
    """Map the public semantic query branch to its extremum direction."""

    if str(query_id) == HIGHEST_FIELD_VALUE_QUERY_ID:
        return "highest"
    if str(query_id) == LOWEST_FIELD_VALUE_QUERY_ID:
        return "lowest"
    raise ValueError(f"unsupported callout metric extremum query_id: {query_id}")


def _build_prompt_slots(target: Dict[str, Any]) -> Dict[str, str]:
    """Bind the resolved metric field and extremum direction into prompt slots."""

    return {
        "field_label": f'"{target["field_label"]}"',
        "rank_direction": str(target["rank_direction"]),
        "rank_order_phrase": str(target["rank_order_phrase"]),
    }


def _metric_extremum_annotation(ctx: Any, callout: Any, target: Dict[str, Any]) -> Dict[str, List[float]]:
    """Collect the winner card plus every compared candidate field row."""

    callout_id = str(callout.callout_id)
    annotation: Dict[str, List[float]] = {
        "winning_callout_card": [float(value) for value in ctx.rendered.callout_bboxes_px[callout_id]],
    }
    for index, candidate in enumerate(target["candidate_values"], start=1):
        candidate_callout = str(candidate["callout_id"])
        candidate_field = str(candidate["field_id"])
        annotation[f"candidate_{index}_field_row"] = [
            float(value) for value in ctx.rendered.field_row_bboxes_px[candidate_callout][candidate_field]
        ]
    return annotation


def _metric_candidate_rank_audit(target: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create a numeric comparison table sorted in the same direction as the prompt."""

    reverse_sort = str(target["rank_direction"]) == "highest"
    rows: List[Dict[str, Any]] = []
    for candidate in target["candidate_values"]:
        rows.append(
            {
                "callout_id": str(candidate["callout_id"]),
                "callout_title": str(candidate["callout_title"]),
                "field_id": str(candidate["field_id"]),
                "visible_value": str(candidate["visible_value"]),
                "numeric_value": int(candidate["numeric_value"]),
                "is_answer": str(candidate["callout_id"]) == str(target["callout_id"]),
            }
        )
    rows.sort(key=lambda row: (int(row["numeric_value"]), str(row["callout_title"])), reverse=reverse_sort)
    return rows


def _metric_extremum_trace_extras(
    field_probs: Dict[str, float],
    target: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Separate query sampling diagnostics from execution-only rank metadata."""

    query_params_extra = {
        "target_field_label_probabilities": dict(field_probs),
    }
    execution_extra = {
        "rank_direction": str(target["rank_direction"]),
        "candidate_rank_audit": _metric_candidate_rank_audit(target),
    }
    return query_params_extra, execution_extra


@register_task
class PagesHeroCalloutMetricExtremumLabelTask:
    """Find the callout title with the highest or lowest visible field value."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Generate one unique-extremum comparison and bind all compared values."""

        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=HIGHEST_FIELD_VALUE_QUERY_ID,
            public_task=TASK_ID,
        )
        task_params = dict(task_params)
        task_params["rank_direction"] = _rank_direction_for_query_id(str(selected_branch))
        ctx = _lifecycle.resolve_scene_context(
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        callout, field, target, field_probs = _lifecycle.select_extremum_target(
            gen_defaults=ctx.gen_defaults,
            spec=ctx.spec,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation = _metric_extremum_annotation(ctx, callout, target)
        query_params_extra, execution_extra = _metric_extremum_trace_extras(field_probs, target)
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
            answer_value=str(target["answer_value"]),
            annotation_type="bbox_map",
            annotation_value=dict(annotation),
            annotation_keys=list(annotation.keys()),
            query_params_extra=query_params_extra,
            execution_extra=execution_extra,
        )
        return _lifecycle.build_string_bbox_map_output(
            ctx=ctx,
            prompt_artifacts=prompt_artifacts,
            answer_value=str(target["answer_value"]),
            annotation=annotation,
            trace_payload=trace_payload,
        )


__all__ = [
    "HIGHEST_FIELD_VALUE_QUERY_ID",
    "LOWEST_FIELD_VALUE_QUERY_ID",
    "PROMPT_QUERY_KEY",
    "QUESTION_FORMAT",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesHeroCalloutMetricExtremumLabelTask",
]
