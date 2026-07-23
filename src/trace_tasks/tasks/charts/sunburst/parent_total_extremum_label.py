"""Public task for `task_charts__sunburst__parent_total_extremum_label`."""

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.sunburst._lifecycle import SunburstTaskPlan, parent_extremum_relations, run_sunburst_task_from_public_class
from trace_tasks.tasks.charts.sunburst.shared.prompts import ANSWER_HINT_LABEL, ANSWER_ONLY_EXAMPLE_LABEL, JSON_EXAMPLE_LABEL, render_prompt_artifacts
from trace_tasks.tasks.charts.sunburst.shared.sampling import descendant_leaf_ids, nodes_by_id, sample_tree, unique_extreme_parent
from trace_tasks.tasks.charts.sunburst.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


PROGRAM_CODE = "arg_extremum(parent, sum(value(leaf) for leaf under parent), direction); output=string_label; annotation=point_set(answer_parent_leaf_value_labels); scene=sunburst; scope=parent_total_extremum_label"


def _direction(selected_query_id):
    if str(selected_query_id) == "highest_parent_total_label":
        return "highest"
    if str(selected_query_id) == "lowest_parent_total_label":
        return "lowest"
    raise ValueError(f"unsupported sunburst parent-total query: {selected_query_id}")


@register_task
class ChartsSunburstParentTotalExtremumLabelTask:
    """Find the parent category with the highest or lowest computed total."""

    task_id = "task_charts__sunburst__parent_total_extremum_label"
    reasoning_operations = ('ranking', 'aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "parent_total_extremum_label"
    supported_query_ids = ("highest_parent_total_label", "lowest_parent_total_label")
    default_query_id = "highest_parent_total_label"
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, params, selected_query_id):
        """Bind all parent totals and annotate the answer parent's leaf values."""

        tree = sample_tree(params, instance_seed=int(instance_seed))
        direction = _direction(selected_query_id)
        answer_parent = unique_extreme_parent(tree, direction=direction)
        lookup = nodes_by_id(tree)
        answer_leaf_ids = descendant_leaf_ids(lookup, str(answer_parent.node_id))
        annotation_leaf_ids = tuple(str(leaf_id) for leaf_id in answer_leaf_ids)
        parent_totals = {str(lookup[parent_id].label): int(lookup[parent_id].value) for parent_id in tree.parent_ids}
        prompt = render_prompt_artifacts(
            prompt_key=str(selected_query_id),
            answer_hint=ANSWER_HINT_LABEL,
            json_example=JSON_EXAMPLE_LABEL,
            json_example_answer_only=ANSWER_ONLY_EXAMPLE_LABEL,
            dynamic_slot_values={},
            instance_seed=int(instance_seed),
        )
        relations = parent_extremum_relations(
            program_code=PROGRAM_CODE,
            direction=direction,
            answer_parent=answer_parent,
            parent_totals=parent_totals,
            answer_leaf_ids=answer_leaf_ids,
            annotation_leaf_ids=annotation_leaf_ids,
        )
        return SunburstTaskPlan(
            tree,
            prompt,
            TypedValue(type="string", value=str(answer_parent.label)),
            str(answer_parent.label),
            "label_open",
            annotation_leaf_ids,
            relations,
            "sunburst_parent_total_extremum",
            {"operation": "arg_extremum", "direction": direction, "parent_totals": parent_totals},
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_sunburst_task_from_public_class(
            self,
            instance_seed=instance_seed,
            params=params,
            max_attempts=max_attempts,
            build_plan=self._build_plan,
        )
