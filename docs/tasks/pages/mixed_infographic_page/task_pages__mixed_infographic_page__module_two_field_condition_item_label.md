# `task_pages__mixed_infographic_page__module_two_field_condition_item_label`

## Identity
1. Domain: `pages`
2. Scene id: `mixed_infographic_page`
3. Source scene: `mixed_infographic_page`
4. Task id: `task_pages__mixed_infographic_page__module_two_field_condition_item_label`

## Contract
1. Objective: find the unique item in one titled module satisfying one numeric threshold condition and one categorical equality condition.
2. Public task contract: `module_two_field_condition_item_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: the visible item row/card/container holding the selected answer item label; condition role boxes and the text-label box are retained in trace diagnostics.
7. Query argument axes: target module, numeric field, categorical field, numeric operator, threshold, categorical value, module count, item/field supports, scene variant, and native layout mode.

## Program Contract
- `module_two_field_condition_item(module_title, numeric_field_label, numeric_condition, category_field_label, category_value); output=string_visible_item_label; annotation=bbox(matching_item_container); scene=mixed_infographic_page; scope=one titled module within one dense mixed infographic page`

## Reasoning Operations

Families: `filtering`, `comparison`

## Prompt + Trace
1. Prompt bundle: `pages_mixed_infographic_page_v1`
2. Scene key: `mixed_infographic_page`
3. Task key: `mixed_infographic_lookup_query`
4. Prompt query key: `module_two_field_condition_item_label`
5. Trace records both conditions, individual condition match sets, their unique intersection, final bboxes, style metadata, and layout geometry.
6. Generation requires each condition alone to leave multiple candidates while their intersection has exactly one item.
7. Explicitly infeasible row layouts and public annotation boxes below the 24-pixel minimum retry with deterministic derived seeds up to `max_attempts`. Trace metadata records the selected attempt index, seed, and measured annotation size; unrelated construction or target-binding errors propagate immediately.
