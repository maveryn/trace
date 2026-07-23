# `task_pages__mixed_infographic_page__module_condition_item_count`

## Identity
1. Domain: `pages`
2. Scene id: `mixed_infographic_page`
3. Source scene: `mixed_infographic_page`
4. Task id: `task_pages__mixed_infographic_page__module_condition_item_count`

## Contract
1. Objective: count items in one titled module whose visible numeric field value satisfies a threshold condition.
2. Public task contract: `module_condition_item_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: unordered value-cell boxes for every matching item; an empty set is not sampled for this task.
7. Query argument axes: target module, numeric field, numeric operator, threshold, module count, item/field supports, scene variant, and native layout mode.

## Program Contract
- `module_condition_item_count(module_title, field_label, numeric_condition); output=integer_count; annotation=bbox_set(matching_value_cells); scene=mixed_infographic_page; scope=one titled module within one dense mixed infographic page`

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Prompt + Trace
1. Prompt bundle: `pages_mixed_infographic_page_v1`
2. Scene key: `mixed_infographic_page`
3. Task key: `mixed_infographic_lookup_query`
4. Prompt query key: `module_condition_item_count`
5. Trace records candidate values, parsed numeric values, threshold/operator, matching values, final bboxes, style metadata, and layout geometry.
6. Threshold sampling keeps the answer nonzero and not all visible items.
7. Explicitly infeasible row layouts and public annotation boxes below the 24-pixel minimum retry with deterministic derived seeds up to `max_attempts`. Trace metadata records the selected attempt index, seed, and measured annotation size; unrelated construction or target-binding errors propagate immediately.
