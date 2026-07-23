"""Public task for `task_charts__curve_panels__cross_panel_delta_extremum_label`."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.unanswerable import (
    UNANSWERABLE_ANSWER,
    absence_proof,
    should_use_unanswerable_branch,
)
from trace_tasks.tasks.charts.curve_panels._lifecycle import (
    CurvePanelTaskPlan,
    build_curve_panel_plan_from_query,
    build_curve_panel_query_record,
    run_curve_panel_task_lifecycle,
)
from trace_tasks.tasks.charts.curve_panels.shared.defaults import (
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.charts.curve_panels.shared.sampling import (
    balanced_choice,
    common_axes,
    make_random_panels,
    palette,
    panel_answer_index_support,
    point_id,
    replace_curve_interval,
    without_sample_cursor,
)

QUERY_ID = "cross_panel_delta_extremum_label"
TASK_PARAM_DEFAULTS: dict[str, Any] = {}
UNANSWERABLE_PROMPT_INSTRUCTION = (
    'If that method is not plotted in every subplot, treat the question as unanswerable '
    'and set the answer to exactly "unanswerable".'
)


@register_task
class ChartsScientificCrossPanelDeltaExtremumLabelTask:
    """Select the panel with the largest increase for one method over an interval."""

    task_id = "task_charts__curve_panels__cross_panel_delta_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = "charts"
    objective_contract = "cross_panel_delta_extremum_label"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True
    supports_unanswerable = True

    def _build_cross_panel_delta_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        answer_panel_index = int(
            balanced_choice(
                panel_answer_index_support(effective_params),
                effective_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.cross_panel_delta.answer",
            )
        )
        non_answer_params = without_sample_cursor(effective_params)
        min_panel_count = max(4, int(answer_panel_index) + 1)
        (
            sampled_x_values,
            y_min,
            y_max,
            _panel_total,
            panel_labels,
            method_labels,
            panel_label_meta,
        ) = common_axes(
            effective_params,
            instance_seed=int(instance_seed),
            min_panel_count=int(min_panel_count),
        )
        answer_panel = str(panel_labels[int(answer_panel_index)])
        colors = palette(effective_params)
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.cross_panel_delta")
        method_label = str(
            balanced_choice(
                method_labels,
                non_answer_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.cross_panel_delta.method",
            )
        )
        max_start_index = max(1, len(sampled_x_values) - 3)
        start_index = int(rng.randint(1, int(max_start_index)))
        end_min = min(len(sampled_x_values) - 2, int(start_index) + 1)
        end_max = len(sampled_x_values) - 2
        if int(end_min) > int(end_max):
            raise RuntimeError("x value support is too small for cross-panel delta")
        end_index = int(rng.randint(int(end_min), int(end_max)))
        start_x = int(sampled_x_values[int(start_index)])
        end_x = int(sampled_x_values[int(end_index)])
        values = make_random_panels(
            panel_labels=panel_labels,
            method_labels=method_labels,
            x_count=len(sampled_x_values),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.cross_panel_delta.values",
            value_min=y_min,
            value_max=y_max,
        )
        if should_use_unanswerable_branch(
            effective_params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.cross_panel_delta",
            enabled=bool(self.supports_unanswerable),
        ):
            missing_panel_candidates = [str(label) for label in panel_labels[1:]]
            if not missing_panel_candidates:
                raise RuntimeError("curve-panel unanswerable branch needs at least two panels")
            missing_panel = str(
                missing_panel_candidates[
                    int(rng.randrange(len(missing_panel_candidates)))
                ]
            )
            present_panels = [
                str(label) for label in panel_labels if str(label) != str(missing_panel)
            ]
            method_presence = {
                str(label): str(label) != str(missing_panel) for label in panel_labels
            }
            missing_trace = {
                "method_label": str(method_label),
                "requested_method_label": str(method_label),
                "visible_method_labels": [str(label) for label in method_labels],
                "missing_method_panel_labels": [str(missing_panel)],
                "present_method_panel_labels": list(present_panels),
                "method_panel_presence": dict(method_presence),
                "start_x_value": int(start_x),
                "end_x_value": int(end_x),
                "deltas_by_panel": {},
                "winning_panel_label": UNANSWERABLE_ANSWER,
                "answerability": "unanswerable",
                "absence_proof": absence_proof(
                    requested_item=f"{str(method_label)} in every subplot",
                    visible_candidates=[
                        f"{str(method_label)} in {str(panel)}"
                        for panel in present_panels
                    ],
                    checked_scope="curve-panel method presence by subplot",
                    absence_reason="requested method is not plotted in every subplot",
                ),
                **dict(panel_label_meta),
            }
            query = build_curve_panel_query_record(
                prompt_key=selected_query_id,
                answer=UNANSWERABLE_ANSWER,
                answer_type="string",
                method_label=str(method_label),
                start_x_value=start_x,
                end_x_value=end_x,
                annotation_panel_labels=(),
                annotation_keyed_point_ids={},
                trace=missing_trace,
            )
            return build_curve_panel_plan_from_query(
                x_values=tuple(sampled_x_values),
                y_min=int(y_min),
                y_max=int(y_max),
                panel_labels=tuple(panel_labels),
                method_labels=tuple(method_labels),
                colors=tuple(colors),
                values_by_panel_method=values,
                query=query,
                dynamic_slots={
                    "method_label": f'"{query.method_label}"',
                    "start_x_value": str(query.start_x_value),
                    "end_x_value": str(query.end_x_value),
                    "unanswerable_instruction": UNANSWERABLE_PROMPT_INSTRUCTION,
                },
                instance_seed=int(instance_seed),
                annotation_type="point_map",
                allow_empty_annotation=True,
                omitted_panel_methods={str(missing_panel): (str(method_label),)},
            )
        target_delta = int(42 + rng.randint(0, 12))
        deltas: Dict[str, int] = {}
        for panel_index, panel in enumerate(panel_labels):
            if str(panel) == str(answer_panel):
                start_value = int(rng.randint(18, 34))
                delta = int(target_delta)
            else:
                start_value = int(rng.randint(24, 56))
                delta = int(rng.randint(-12, int(target_delta) - 14))
            end_value = max(
                int(y_min) + 5, min(int(y_max) - 5, int(start_value) + int(delta))
            )
            actual_delta = int(end_value) - int(start_value)
            if str(panel) != str(answer_panel) and int(actual_delta) >= int(
                target_delta
            ):
                end_value = (
                    int(start_value) + int(target_delta) - 14 - (int(panel_index) % 5)
                )
                end_value = max(int(y_min) + 5, min(int(y_max) - 5, int(end_value)))
                actual_delta = int(end_value) - int(start_value)
            deltas[str(panel)] = int(actual_delta)
            replace_curve_interval(
                values[str(panel)][str(method_label)],
                start_index=int(start_index),
                end_index=int(end_index),
                start_value=int(start_value),
                end_value=int(end_value),
                rng=rng,
                y_min=int(y_min),
                y_max=int(y_max),
            )
        if max(deltas, key=lambda label: (deltas[label], label)) != str(answer_panel):
            raise RuntimeError("cross-panel delta construction lost unique target")
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=answer_panel,
            answer_type="string",
            panel_label=answer_panel,
            method_label=method_label,
            start_x_value=start_x,
            end_x_value=end_x,
            annotation_panel_labels=(answer_panel,),
            annotation_keyed_point_ids={
                "start_point": point_id(answer_panel, method_label, int(start_x)),
                "end_point": point_id(answer_panel, method_label, int(end_x)),
            },
            trace={
                "method_label": str(method_label),
                "start_x_value": int(start_x),
                "end_x_value": int(end_x),
                "deltas_by_panel": dict(deltas),
                "winning_panel_label": str(answer_panel),
                "answerability": "answerable",
                **dict(panel_label_meta),
            },
        )
        return build_curve_panel_plan_from_query(
            x_values=tuple(sampled_x_values),
            y_min=int(y_min),
            y_max=int(y_max),
            panel_labels=tuple(panel_labels),
            method_labels=tuple(method_labels),
            colors=tuple(colors),
            values_by_panel_method=values,
            query=query,
            dynamic_slots={
                "method_label": f'"{query.method_label}"',
                "start_x_value": str(query.start_x_value),
                "end_x_value": str(query.end_x_value),
                "unanswerable_instruction": UNANSWERABLE_PROMPT_INSTRUCTION,
            },
            instance_seed=int(instance_seed),
            annotation_type="point_map",
        )

    def generate(
        self, instance_seed: int, *, params: dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Select the local query, then run neutral curve-panel lifecycle."""

        return run_curve_panel_task_lifecycle(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            supported_query_ids=self.supported_query_ids,
            default_query_id=QUERY_ID,
            failure_label=self.task_id,
            build_plan=self._build_cross_panel_delta_plan,
        )


__all__ = ["ChartsScientificCrossPanelDeltaExtremumLabelTask"]
