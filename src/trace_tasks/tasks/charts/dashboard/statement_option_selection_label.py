"""Public task for `task_charts__dashboard__statement_option_selection_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import DashboardTaskPlan, MaterializedDashboardTask, dashboard_task_output_fields, run_dashboard_public_task
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import build_dashboard_base_sample, sample_panel_title_labels
from trace_tasks.tasks.charts.dashboard.shared.state import DOMAIN, SCENE_ID, SCENE_VARIANT, OPTION_LETTERS, DashboardDataset, DashboardQuery
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))
from trace_tasks.tasks.charts.dashboard.shared.metrics import balanced_support_choice, option_count_support, statement_option_candidates, weighted_choice_from_defaults
from trace_tasks.tasks.charts.dashboard.shared.state import SUPPORTED_REQUESTED_TRUTHS


QUERY_ID = "statement_option_selection_label"
TASK_PARAM_DEFAULTS: dict[str, Any] = {"canvas_width": 1200, "canvas_height": 1064, "category_count_max": 8, "context_text_enabled": False, "option_panel_height_px": 178, "option_panel_gap_px": 16, "option_panel_padding_px": 16, "option_panel_column_count": 2, "option_panel_column_gap_px": 22, "option_panel_font_size_px": 15, "option_panel_letter_font_size_px": 16}


@register_task
class ChartsDashboardStatementOptionSelectionLabelTask:
    """Select the statement option whose truth value matches the requested condition."""

    task_id = "task_charts__dashboard__statement_option_selection_label"
    reasoning_operations = ('matching',)
    domain = DOMAIN
    objective_contract = "statement_option_selection_label"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def _prepare_statement_option_plan(self, instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
        """Bind one requested truth option and preserve its two chart witnesses."""
        del selected_query_id
        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{self.objective_contract}.selection")
        base_sample = build_dashboard_base_sample(effective_params, instance_seed=int(instance_seed))
        option_counts = option_count_support(effective_params)
        explicit_answer_letter = effective_params.get("answer_letter")
        if explicit_answer_letter is not None:
            answer_letter = str(explicit_answer_letter).upper()
            if answer_letter not in OPTION_LETTERS:
                raise ValueError("answer_letter must be A..F")
        else:
            feasible_letters = OPTION_LETTERS[: max(option_counts)]
            answer_letter = str(
                uniform_choice(
                    spawn_rng(
                        int(instance_seed),
                        f"{SCENE_ID}.statement_option.answer_letter",
                    ),
                    feasible_letters,
                )
            )
        feasible_option_counts = tuple(int(count) for count in option_counts if OPTION_LETTERS.index(str(answer_letter)) < int(count))
        if not feasible_option_counts:
            raise ValueError("answer_letter is infeasible for configured option_count support")
        option_count = balanced_support_choice(params=effective_params, instance_seed=int(instance_seed), namespace=f"{SCENE_ID}.statement_option.option_count.{answer_letter}", support=feasible_option_counts)
        requested_truth = weighted_choice_from_defaults(rng, params=effective_params, key="requested_truth", supported=SUPPORTED_REQUESTED_TRUTHS, fallback_weights_key="statement_requested_truth_weights")
        target_truth = str(requested_truth) == "true"
        candidates = statement_option_candidates(rng=rng, categories=base_sample.categories, panels=base_sample.panels)
        selected_pool = [candidate for candidate in candidates if bool(candidate["truth_value"]) is bool(target_truth)]
        distractor_pool = [candidate for candidate in candidates if bool(candidate["truth_value"]) is not bool(target_truth)]
        if not selected_pool or len(distractor_pool) < int(option_count) - 1:
            raise ValueError("not enough dashboard statement candidates for requested option set")
        selected = dict(selected_pool[int(rng.randrange(len(selected_pool)))])
        distractors: list[dict[str, Any]] = []
        used_texts = {str(selected["text"])}
        for candidate in distractor_pool:
            if str(candidate["text"]) in used_texts:
                continue
            distractors.append(dict(candidate))
            used_texts.add(str(candidate["text"]))
            if len(distractors) >= int(option_count) - 1:
                break
        if len(distractors) != int(option_count) - 1:
            raise ValueError("failed to construct unique statement distractors")
        option_letters = OPTION_LETTERS[: int(option_count)]
        answer_index = option_letters.index(str(answer_letter))
        option_records: list[dict[str, Any]] = []
        distractor_index = 0
        for index, option_label in enumerate(option_letters):
            record = dict(selected) if int(index) == int(answer_index) else dict(distractors[distractor_index])
            if int(index) != int(answer_index):
                distractor_index += 1
            record["option_label"] = str(option_label)
            record["option_id"] = f"option_{option_label}"
            option_records.append(record)
        selected_record = option_records[int(answer_index)]
        matching_labels = [str(option["option_label"]) for option in option_records if bool(option["truth_value"]) is bool(target_truth)]
        if matching_labels != [str(answer_letter)]:
            raise ValueError("statement option set does not have exactly one requested truth match")
        relations = {
            **dict(base_sample.common_params),
            "requested_truth": str(requested_truth),
            "requested_truth_phrase": "true" if bool(target_truth) else "false",
            "option_count": int(option_count),
            "option_labels": list(option_letters),
            "answer_option_label": str(answer_letter),
            "statement_options": [dict(option) for option in option_records],
            "selected_statement": dict(selected_record),
        }
        refs = ((str(selected_record["first_panel_id"]), str(selected_record["first_category_id"])), (str(selected_record["second_panel_id"]), str(selected_record["second_category_id"])))
        dataset = DashboardDataset(scene_variant=SCENE_VARIANT, categories=base_sample.categories, panels=base_sample.panels, query=DashboardQuery(answer=str(answer_letter), answer_type="option_letter", annotation_refs=refs, params=dict(relations)))
        prompt_artifacts = build_prompt_artifacts(prompt_query_key=QUERY_ID, dynamic_slots=build_prompt_slots(dataset=dataset), instance_seed=int(instance_seed))
        return DashboardTaskPlan(dataset=dataset, prompt_artifacts=prompt_artifacts, relations=relations, answer_gt=TypedValue(type="option_letter", value=str(answer_letter)), annotation_refs=refs)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params={**TASK_PARAM_DEFAULTS, **dict(params)}, supported_query_ids=self.supported_query_ids, default_query_id=QUERY_ID, task_id=self.task_id)
        return run_dashboard_public_task(instance_seed=int(instance_seed), params=task_params, max_attempts=int(max_attempts), selected_query_id=str(selected_query_id), build_plan=self._prepare_statement_option_plan, build_output=_build_task_output)


__all__ = ["ChartsDashboardStatementOptionSelectionLabelTask"]
