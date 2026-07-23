# `task_pages__control_board__state_extremum_group_label`

## Identity
1. Domain: `pages`
2. Scene id: `control_board`
3. Source scene: `control_board`
4. Task id: `task_pages__control_board__state_extremum_group_label`

## Contract
1. Objective: identify the control group with the unique highest count of controls matching one requested visual state condition.
2. Public task contract: `state_extremum_group_label`
3. Supported `query_id` values: `disabled_extremum_group_label`, `selected_enabled_extremum_group_label`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: the full panel box for the group that has the most matching controls.
7. Query argument axes: state condition, target group position, target state count, scene variant, and visual style.

## Program Contract
- `argmax_group(count(filter(gui_controls, group_name=group and state_condition in {enabled=false, selected=true and enabled=true}))); output=group_name_string; annotation=bbox(selected_group_panel); scene=control_board; scope=one grouped control-board screen`

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `logical_composition`

## Prompt + Trace
1. Prompt bundle: `pages_control_board_v1`
2. Scene key: `control_board`
3. Task key: `control_board_count_query`
4. Prompt query keys: `disabled_extremum_group_label`, `selected_enabled_extremum_group_label`
5. Trace records control groups, per-control state flags, per-group matching-state counts, final group/control bboxes, selected query id, and selected target metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized control-board render metadata.
