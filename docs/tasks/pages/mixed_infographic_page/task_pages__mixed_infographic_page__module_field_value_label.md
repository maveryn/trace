# `task_pages__mixed_infographic_page__module_field_value_label`

## Identity
1. Domain: `pages`
2. Scene id: `mixed_infographic_page`
3. Source scene: `mixed_infographic_page`
4. Task id: `task_pages__mixed_infographic_page__module_field_value_label`

## Contract
1. Objective: rank items in one titled module by a selector field, then read a second field value for the ranked item.
2. Public task contract: `module_field_value_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: the answer `value_cell`; lookup role boxes are retained in trace diagnostics.
7. Query argument axes: target module, selector field, rank direction, rank position, answer field, module count, item/field supports, scene variant, and native layout mode.

## Program Contract
- `ranked_module_field_value(module_title, selector_field_label, rank_direction, rank_position, answer_field_label); output=string_visible_value; annotation=bbox(answer_value_cell); scene=mixed_infographic_page; scope=one dense mixed infographic page`

## Reasoning Operations

Families: `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_mixed_infographic_page_v1`
2. Scene key: `mixed_infographic_page`
3. Task key: `mixed_infographic_lookup_query`
4. Prompt query key: `module_field_value_label`
5. Trace records module/item/selector-field/answer-field ids, visible labels and values, final bboxes, style metadata, and layout geometry.
6. Generation is deterministic from `instance_seed`; answers and annotation come from finalized render metadata.
7. Explicitly infeasible row layouts and public annotation boxes below the 24-pixel minimum retry with deterministic derived seeds up to `max_attempts`. Trace metadata records the selected attempt index, seed, and measured annotation size; unrelated construction or target-binding errors propagate immediately.
