"""Public task for `task_charts__sunburst__leaf_threshold_count_under_parent`."""

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.sunburst._lifecycle import SunburstTaskPlan, leaf_condition_relations, run_sunburst_task_from_public_class
from trace_tasks.tasks.charts.sunburst.shared.prompts import ANSWER_HINT_INTEGER, ANSWER_ONLY_EXAMPLE_COUNT, JSON_EXAMPLE_COUNT, render_prompt_artifacts
from trace_tasks.tasks.charts.sunburst.shared.sampling import nodes_by_id, sample_tree, threshold_leaf_case, threshold_matching_leaf_ids
from trace_tasks.tasks.charts.sunburst.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


PROGRAM_CODE = "count(leaf under parent where compare(value(leaf), threshold, direction)); output=integer_count; annotation=point_set(matching_leaf_value_labels); scene=sunburst; scope=leaf_threshold_count_under_parent"


def _comparator(selected_query_id):
    if str(selected_query_id) == "above_threshold_leaf_count_under_parent":
        return "above"
    if str(selected_query_id) == "below_threshold_leaf_count_under_parent":
        return "below"
    raise ValueError(f"unsupported sunburst threshold query: {selected_query_id}")


@register_task
class ChartsSunburstLeafThresholdCountUnderParentTask:
    """Count outer leaves under a parent that satisfy a one-bound threshold."""

    task_id = "task_charts__sunburst__leaf_threshold_count_under_parent"
    reasoning_operations = ('filtering', 'counting', 'comparison', 'topology')
    domain = DOMAIN
    objective_contract = "leaf_threshold_count_under_parent"
    supported_query_ids = ("above_threshold_leaf_count_under_parent", "below_threshold_leaf_count_under_parent")
    default_query_id = "above_threshold_leaf_count_under_parent"
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, params, selected_query_id):
        """Bind threshold direction, counted leaves, and matching value boxes."""

        tree = sample_tree(params, instance_seed=int(instance_seed))
        case = threshold_leaf_case(
            tree,
            comparator=_comparator(selected_query_id),
            params=params,
            instance_seed=int(instance_seed),
        )
        matching_leaf_ids = threshold_matching_leaf_ids(case, nodes_by_id(tree))
        prompt = render_prompt_artifacts(
            prompt_key=str(selected_query_id),
            answer_hint=ANSWER_HINT_INTEGER,
            json_example=JSON_EXAMPLE_COUNT,
            json_example_answer_only=ANSWER_ONLY_EXAMPLE_COUNT,
            dynamic_slot_values={
                "parent_label": str(case["parent_label"]),
                "threshold_value": str(case["threshold_value"]),
            },
            instance_seed=int(instance_seed),
        )
        relations = leaf_condition_relations(
            program_code=PROGRAM_CODE,
            case=case,
            fields={
                "comparison_phrase": str(case["comparison_phrase"]),
                "threshold_value": int(case["threshold_value"]),
            },
        )
        return SunburstTaskPlan(
            tree,
            prompt,
            TypedValue(type="integer", value=int(case["answer"])),
            int(case["answer"]),
            "numeric_open",
            matching_leaf_ids,
            relations,
            "sunburst_leaf_threshold_count",
            {
                "operation": "threshold_count",
                "comparison": str(case["comparison_phrase"]),
                "threshold_value": int(case["threshold_value"]),
                "parent_label": str(case["parent_label"]),
            },
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_sunburst_task_from_public_class(
            self,
            instance_seed=instance_seed,
            params=params,
            max_attempts=max_attempts,
            build_plan=self._build_plan,
        )
