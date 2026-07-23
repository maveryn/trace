# `task_pages__mixed_infographic_page__module_field_total_value`

## Identity
1. Domain: `pages`
2. Scene id: `mixed_infographic_page`
3. Source scene: `mixed_infographic_page`
4. Task id: `task_pages__mixed_infographic_page__module_field_total_value`

## Contract
1. Objective: sum all visible values for one additive numeric field inside one titled module.
2. Public task contract: `module_field_total_value`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: unordered value-cell boxes for every value included in the sum.
7. Query argument axes: target module, additive numeric field, module count, item/field supports, scene variant, and native layout mode.

## Program Contract
- `module_field_total(module_title, field_label); output=integer_sum; annotation=bbox_set(summed_value_cells); scene=mixed_infographic_page; scope=one titled module within one dense mixed infographic page`

## Reasoning Operations

Families: `aggregation`

## Prompt + Trace
1. Prompt bundle: `pages_mixed_infographic_page_v1`
2. Scene key: `mixed_infographic_page`
3. Task key: `mixed_infographic_lookup_query`
4. Prompt query key: `module_field_total_value`
5. Trace records the summed item values, parsed numeric values, final bboxes, style metadata, and layout geometry.
6. Additive fields are limited to `Score` and `Count`.
7. Explicitly infeasible row layouts and public annotation boxes below the 24-pixel minimum retry with deterministic derived seeds up to `max_attempts`. Trace metadata records the selected attempt index, seed, and measured annotation size; unrelated construction or target-binding errors propagate immediately.
