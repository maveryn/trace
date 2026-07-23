# `task_pages__mixed_infographic_page__two_module_field_total_comparison_module_label`

## Identity
1. Domain: `pages`
2. Scene id: `mixed_infographic_page`
3. Source scene: `mixed_infographic_page`
4. Task id: `task_pages__mixed_infographic_page__two_module_field_total_comparison_module_label`

## Contract
1. Objective: compare the totals for one shared additive field across two titled modules and return the module title with the larger total.
2. Public task contract: `two_module_field_total_comparison_module_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: the winning module panel; summed value-cell boxes are retained in trace diagnostics.
7. Query argument axes: shared additive field, target module pair, module count, item/field supports, scene variant, and native layout mode.

## Program Contract
- `two_module_field_total_comparison(module_a_title, module_b_title, field_label); output=string_visible_module_title; annotation=bbox(winning_module_panel); scene=mixed_infographic_page; scope=two titled modules within one dense mixed infographic page`

## Reasoning Operations

Families: `ranking`, `aggregation`

## Prompt + Trace
1. Prompt bundle: `pages_mixed_infographic_page_v1`
2. Scene key: `mixed_infographic_page`
3. Task key: `mixed_infographic_lookup_query`
4. Prompt query key: `two_module_field_total_comparison_module_label`
5. Trace records both module titles, the shared field, summed values, computed totals, winning side, final bboxes, style metadata, and layout geometry.
6. Generation guarantees that both modules contain the shared additive field and that the two totals are unequal.
7. Explicitly infeasible row layouts and public annotation boxes below the 24-pixel minimum retry with deterministic derived seeds up to `max_attempts`. Trace metadata records the selected attempt index, seed, and measured annotation size; unrelated construction or target-binding errors propagate immediately.
