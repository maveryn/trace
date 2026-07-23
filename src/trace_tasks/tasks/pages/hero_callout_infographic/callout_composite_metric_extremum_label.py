"""Hero-callout task for selecting the extreme callout by a two-field sum."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__hero_callout_infographic__callout_composite_metric_extremum_label"
HIGHEST_COMPOSITE_QUERY_ID = "highest_composite_metric_callout_label"
LOWEST_COMPOSITE_QUERY_ID = "lowest_composite_metric_callout_label"
SUPPORTED_QUERY_IDS = (HIGHEST_COMPOSITE_QUERY_ID, LOWEST_COMPOSITE_QUERY_ID)
PROMPT_QUERY_KEY = "callout_composite_metric_extremum_label"
QUESTION_FORMAT = "hero_callout_infographic_callout_composite_metric_extremum_label"
COMPOSITE_FIELD_LABELS = ("Score", "Count")


def _rank_direction_for_query_id(query_id: str) -> str:
    """Map this task's public branch to the requested rank direction."""

    if str(query_id) == HIGHEST_COMPOSITE_QUERY_ID:
        return "highest"
    if str(query_id) == LOWEST_COMPOSITE_QUERY_ID:
        return "lowest"
    raise ValueError(f"unsupported callout composite metric query_id: {query_id}")


@register_task
class PagesHeroCalloutCompositeMetricExtremumLabelTask:
    """Find the callout title with the highest or lowest two-field sum."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation')
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        """Generate a unique two-field sum extremum and bind all compared rows."""

        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=HIGHEST_COMPOSITE_QUERY_ID,
            public_task=TASK_ID,
        )
        task_params = dict(task_params)
        task_params["rank_direction"] = _rank_direction_for_query_id(str(selected_branch))
        task_params.setdefault("target_field_labels", COMPOSITE_FIELD_LABELS)
        task_params.setdefault("required_field_labels", COMPOSITE_FIELD_LABELS)
        task_params.setdefault("field_count_support", (3,))
        task_params.setdefault("callout_count_support", (5, 6, 7))
        ctx = _lifecycle.resolve_scene_context(
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        callout, target, field_pair_probs = _lifecycle.select_composite_extremum_target(
            spec=ctx.spec,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation = _lifecycle._composite_extremum_annotation(ctx=ctx, callout=callout, target=target)
        prompt_artifacts = _lifecycle.render_prompt(
            prompt_query_key=PROMPT_QUERY_KEY,
            dynamic_slots=_lifecycle._composite_prompt_slots(target),
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
            query_params_extra={
                "target_field_pair_probabilities": dict(field_pair_probs),
            },
            execution_extra={
                "rank_direction": str(target["rank_direction"]),
                "candidate_rank_audit": _lifecycle._composite_rank_audit(target),
            },
        )
        return _lifecycle.build_string_bbox_map_output(
            ctx=ctx,
            prompt_artifacts=prompt_artifacts,
            answer_value=str(target["answer_value"]),
            annotation=annotation,
            trace_payload=trace_payload,
        )


__all__ = [
    "COMPOSITE_FIELD_LABELS",
    "HIGHEST_COMPOSITE_QUERY_ID",
    "LOWEST_COMPOSITE_QUERY_ID",
    "PROMPT_QUERY_KEY",
    "QUESTION_FORMAT",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesHeroCalloutCompositeMetricExtremumLabelTask",
]
