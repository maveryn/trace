# `task_pages__record_table__value_threshold_in_group_count`

## Identity
1. Domain: `pages`
2. Scene id: `record_table`
3. Source scene: `record_table`
4. Task id: `task_pages__record_table__value_threshold_in_group_count`

## Contract
1. Objective: count visible rows in the requested table section whose Size is at least the requested threshold.
2. Public task contract: `value_threshold_in_group_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: full row boxes for every counted row in the requested section meeting the threshold.
7. Query argument axes: target section name, size threshold, answer-count support, scene variant, and style variant.

## Program Contract
- `record_table_value_threshold_in_group_count(section_name, size_threshold_mb); output=integer_value; annotation=bbox_set(counted_rows); scene=record_table; scope=one sectioned record-table page`

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Prompt + Trace
1. Prompt bundle: `pages_record_table_v1`
2. Scene key: `record_table`
3. Task key: `record_table_count_query`
4. Prompt query key: `value_threshold_in_group_count`
5. Trace records section membership, row Size values, final row bboxes, sampled scene/style metadata, and selected target metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized record-table render metadata.
