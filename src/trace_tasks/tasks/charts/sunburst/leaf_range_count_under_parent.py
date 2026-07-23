"""Public task for `task_charts__sunburst__leaf_range_count_under_parent`."""

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.sunburst._lifecycle import SunburstTaskPlan, leaf_condition_relations, run_sunburst_task_from_public_class
from trace_tasks.tasks.charts.sunburst.shared.prompts import ANSWER_HINT_INTEGER, ANSWER_ONLY_EXAMPLE_COUNT, JSON_EXAMPLE_COUNT, render_prompt_artifacts
from trace_tasks.tasks.charts.sunburst.shared.sampling import nodes_by_id, range_leaf_case, range_matching_leaf_ids, sample_tree
from trace_tasks.tasks.charts.sunburst.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


PROGRAM_CODE = "count(leaf under parent where lower <= value(leaf) <= upper); output=integer_count; annotation=point_set(matching_leaf_value_labels); scene=sunburst; scope=leaf_range_count_under_parent"


@register_task
class ChartsSunburstLeafRangeCountUnderParentTask:
    """Count outer leaves under a parent whose values fall inside a range."""

    task_id = "task_charts__sunburst__leaf_range_count_under_parent"
    reasoning_operations = ('filtering', 'counting', 'comparison', 'topology')
    domain = DOMAIN
    objective_contract = "leaf_range_count_under_parent"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, params, selected_query_id):
        """Bind the inclusive value range and every matching leaf-value witness."""

        tree = sample_tree(params, instance_seed=int(instance_seed))
        case = range_leaf_case(tree, params=params, instance_seed=int(instance_seed))
        matching_leaf_ids = range_matching_leaf_ids(case, nodes_by_id(tree))
        prompt = render_prompt_artifacts(
            prompt_key="leaf_range_count_under_parent",
            answer_hint=ANSWER_HINT_INTEGER,
            json_example=JSON_EXAMPLE_COUNT,
            json_example_answer_only=ANSWER_ONLY_EXAMPLE_COUNT,
            dynamic_slot_values={
                "parent_label": str(case["parent_label"]),
                "lower_value": str(case["lower_value"]),
                "upper_value": str(case["upper_value"]),
            },
            instance_seed=int(instance_seed),
        )
        relations = leaf_condition_relations(
            program_code=PROGRAM_CODE,
            case=case,
            fields={
                "lower_value": int(case["lower_value"]),
                "upper_value": int(case["upper_value"]),
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
            "sunburst_leaf_range_count",
            {
                "operation": "inclusive_range_count",
                "lower_value": int(case["lower_value"]),
                "upper_value": int(case["upper_value"]),
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
