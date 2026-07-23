# `task_pages__workspace__context_control_count`

## Identity
1. Domain: `pages`
2. Scene id: `workspace`
3. Source scene: `workspace`
4. Task id: `task_pages__workspace__context_control_count`

## Contract
1. Objective: count controls in one named workspace context row that have a requested visible state.
2. Public task contract: `context_control_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: boxes around the counted controls; empty list when the answer is `0`.
7. Query argument axes: target context row, target visible state, answer count, workspace variant, and style variant.

## Program Contract
- `workspace_context_control_count(context_row, visible_control_state); output=integer_count; annotation=bbox_set(matching_controls); scene=workspace; scope=one professional application workspace`

## Reasoning Operations

Families: `counting`

## Prompt + Trace
1. Prompt bundle: `pages_workspace_v1`
2. Scene key: `workspace`
3. Task key: `workspace_control_query`
4. Prompt query key: `row_state_filter_count`
5. Runtime `query_id` is `single`; workspace flavor and target state metadata are recorded in trace.
6. Trace records all controls, visual state assignments, counted control ids, context row support metadata, sampled workspace/app/style metadata, and prompt metadata.
