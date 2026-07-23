# `task_pages__category_grid__category_slot_item_label`

## Identity
1. Domain: `pages`
2. Scene id: `category_grid`
3. Source scene: `category_grid`
4. Task id: `task_pages__category_grid__category_slot_item_label`

## Contract
1. Objective: find a named category and subcategory, then read the item label at a requested ordinal slot.
2. Public task contract: `category_slot_item_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox_map`
6. Annotation witness: keyed boxes for `category_header`, `subcategory_header`, and `target_item`.
7. Query argument axes: target category, target subcategory, target slot, category count, subcategory count, item-count support, and scene layout variant.

## Program Contract
- `category_grid_slot_item_lookup(category_label, subcategory_label, ordinal_slot); output=item_label_string; annotation=bbox_map(category_header,subcategory_header,target_item); scene=category_grid; scope=one category-grid page`

## Reasoning Operations

Families: `direct_retrieval`

## Prompt + Trace
1. Prompt bundle: `pages_category_grid_v1`
2. Scene key: `category_grid`
3. Task key: `category_grid_lookup_query`
4. Prompt query key: `category_slot_item_label`
5. Trace records category headers, subcategory headers, item order, item labels, final bboxes, sampled style metadata, and layout geometry.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized category-grid render metadata.
