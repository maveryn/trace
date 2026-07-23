# `task_pages__control_board__control_state_condition_count`

## Identity
1. Domain: `pages`
2. Scene id: `control_board`
3. Source scene: `control_board`
4. Task id: `task_pages__control_board__control_state_condition_count`

## Contract
1. Objective: count controls inside one requested control group that match the requested visual state condition.
2. Public task contract: `control_state_condition_count`
3. Supported `query_id` values: `disabled_controls_in_group_count`, `selected_enabled_controls_in_group_count`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: full control boxes for every counted control in the requested group.
7. Query argument axes: state condition, target group name, answer-count support, scene variant, and style variant.

## Program Contract
- `count(filter(gui_controls, group_name=target_group and state_condition in {enabled=false, selected=true and enabled=true})); output=integer_value; annotation=bbox_set(counted_controls); scene=control_board; scope=one grouped control-board screen`

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Prompt + Trace
1. Prompt bundle: `pages_control_board_v1`
2. Scene key: `control_board`
3. Task key: `control_board_count_query`
4. Prompt query keys: `disabled_controls_in_group_count`, `selected_enabled_controls_in_group_count`
5. Trace records control groups, control state flags, command labels, final bboxes, sampled scene/style metadata, selected query id, and selected target metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized control-board render metadata.
