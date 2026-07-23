# `task_pages__mixed_infographic_page__page_field_extremum_module_label`

## Identity
1. Domain: `pages`
2. Scene id: `mixed_infographic_page`
3. Source scene: `mixed_infographic_page`
4. Task id: `task_pages__mixed_infographic_page__page_field_extremum_module_label`

## Contract
1. Objective: find the module containing the highest or lowest visible value for one shared numeric field across the page.
2. Public task contract: `page_field_extremum_module_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: the winning module panel; compared value boxes are retained in trace diagnostics.
7. Query argument axes: shared numeric field, rank direction, module count, item/field supports, scene variant, and native layout mode.

## Program Contract
- `page_field_extremum_module(field_label, rank_direction); output=string_visible_module_title; annotation=bbox(winning_module_panel); scene=mixed_infographic_page; scope=all modules on one dense mixed infographic page that show the shared field`

## Reasoning Operations

Families: `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_mixed_infographic_page_v1`
2. Scene key: `mixed_infographic_page`
3. Task key: `mixed_infographic_lookup_query`
4. Prompt query key: `page_field_extremum_module_label`
5. Trace records the shared field, compared page-wide item values, parsed numeric values, winning item/module, final bboxes, style metadata, and layout geometry.
6. Generation guarantees a unique page-wide extremum.
7. Explicitly infeasible row layouts and public annotation boxes below the 24-pixel minimum retry with deterministic derived seeds up to `max_attempts`. Trace metadata records the selected attempt index, seed, and measured annotation size; unrelated construction or target-binding errors propagate immediately.
