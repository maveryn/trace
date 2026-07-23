# `task_pages__record_table__enabled_action_for_type_count`

## Identity
1. Domain: `pages`
2. Scene id: `record_table`
3. Source scene: `record_table`
4. Task id: `task_pages__record_table__enabled_action_for_type_count`

## Contract
1. Objective: count visible record-table rows whose Type matches the requested label and whose requested action button is enabled.
2. Public task contract: `enabled_action_for_type_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: full row boxes for every counted matching row.
7. Query argument axes: target type label, target action label, answer-count support, scene variant, and style variant.

## Program Contract
- `record_table_enabled_action_for_type_count(type_label, action_label); output=integer_value; annotation=bbox_set(counted_rows); scene=record_table; scope=one sectioned record-table page`

## Reasoning Operations

Families: `filtering`, `counting`

## Prompt + Trace
1. Prompt bundle: `pages_record_table_v1`
2. Scene key: `record_table`
3. Task key: `record_table_count_query`
4. Prompt query key: `enabled_action_for_type_count`
5. Trace records row type labels, action labels, action enabled state, final row bboxes, sampled scene/style metadata, and selected target metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized record-table render metadata.
