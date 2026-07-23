# `task_pages__category_grid__category_item_count`

## Identity
1. Domain: `pages`
2. Scene id: `category_grid`
3. Source scene: `category_grid`
4. Task id: `task_pages__category_grid__category_item_count`

## Contract
1. Objective: count visible item rows inside a requested category and subcategory block.
2. Public task contract: `category_item_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: counted item-row boxes.
7. Query argument axes: target category, target subcategory, category count, subcategory count, item-count support, and scene layout variant.

## Program Contract
- `category_grid_item_count(category_label, subcategory_label); output=integer_value; annotation=bbox_set(counted_item_rows); scene=category_grid; scope=one category-grid page`

## Reasoning Operations

Families: `counting`

## Prompt + Trace
1. Prompt bundle: `pages_category_grid_v1`
2. Scene key: `category_grid`
3. Task key: `category_grid_lookup_query`
4. Prompt query key: `category_item_count`
5. Trace records category headers, subcategory headers, item order, item labels, final bboxes, sampled style metadata, and layout geometry.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized category-grid render metadata.
