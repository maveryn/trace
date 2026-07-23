"""Public task for `task_charts__sunburst__parent_total_value`."""

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.sunburst._lifecycle import SunburstTaskPlan, parent_total_relations, run_sunburst_task_from_public_class
from trace_tasks.tasks.charts.sunburst.shared.prompts import ANSWER_HINT_INTEGER, ANSWER_ONLY_EXAMPLE_TOTAL, JSON_EXAMPLE_TOTAL, render_prompt_artifacts
from trace_tasks.tasks.charts.sunburst.shared.sampling import choose_parent, descendant_leaf_ids, nodes_by_id, sample_tree
from trace_tasks.tasks.charts.sunburst.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


PROGRAM_CODE = "sum(value(leaf) for leaf under parent); output=integer_value; annotation=point_set(leaf_value_labels); scene=sunburst; scope=parent_total_value"


@register_task
class ChartsSunburstParentTotalValueTask:
    """Compute a parent total from outer leaf values in a sunburst hierarchy."""

    task_id = "task_charts__sunburst__parent_total_value"
    reasoning_operations = ('aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "parent_total_value"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, params, selected_query_id):
        """Bind the requested parent and its descendant leaf-value witnesses."""

        tree = sample_tree(params, instance_seed=int(instance_seed))
        parent = choose_parent(tree, instance_seed=int(instance_seed), namespace="charts.sunburst.parent_total.parent")
        lookup = nodes_by_id(tree)
        leaf_ids = descendant_leaf_ids(lookup, str(parent.node_id))
        leaf_values = [int(lookup[str(leaf_id)].value) for leaf_id in leaf_ids]
        prompt = render_prompt_artifacts(
            prompt_key="parent_total_from_leaves_value",
            answer_hint=ANSWER_HINT_INTEGER,
            json_example=JSON_EXAMPLE_TOTAL,
            json_example_answer_only=ANSWER_ONLY_EXAMPLE_TOTAL,
            dynamic_slot_values={"parent_label": str(parent.label)},
            instance_seed=int(instance_seed),
        )
        relations = parent_total_relations(
            program_code=PROGRAM_CODE,
            parent=parent,
            leaf_ids=leaf_ids,
            leaf_values=leaf_values,
        )
        return SunburstTaskPlan(
            tree,
            prompt,
            TypedValue(type="integer", value=int(parent.value)),
            int(parent.value),
            "numeric_open",
            tuple(str(leaf_id) for leaf_id in leaf_ids),
            relations,
            "sunburst_parent_total",
            {"operation": "sum", "parent_label": str(parent.label), "leaf_values": leaf_values},
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_sunburst_task_from_public_class(
            self,
            instance_seed=instance_seed,
            params=params,
            max_attempts=max_attempts,
            build_plan=self._build_plan,
        )
