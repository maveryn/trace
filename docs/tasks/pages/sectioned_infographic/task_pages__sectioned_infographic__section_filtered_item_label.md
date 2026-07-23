# `task_pages__sectioned_infographic__section_filtered_item_label`

## Identity
1. Domain: `pages`
2. Scene id: `sectioned_infographic`
3. Source scene: `sectioned_infographic`
4. Task id: `task_pages__sectioned_infographic__section_filtered_item_label`

## Program Contract
1. Program schema: `sectioned_infographic_section_filtered_item_label(section_title, marker_label) -> item_label; scene=sectioned_infographic; scope=section_filtered_item_label`
2. Scene: `sectioned_infographic`
3. Scope: one rendered sectioned infographic with named sections, visible marker shapes, and visible item rows.
4. Supported `query_id`: `single`
5. Answer schema: `string`
6. Annotation schema: `bbox`
7. Annotation witness: one box around the matching item row.
8. Query arguments: resolved target section title and unique marker label within that section.
9. Render arguments: section count, per-section item-count support, target marker, scene layout variant, visual style, and post-render noise.

## Reasoning Operations

Families: `filtering`

## Prompt + Trace
1. Prompt bundle: `pages_sectioned_infographic_v1`
2. Scene key: `sectioned_infographic`
3. Task key: `sectioned_infographic_query`
4. Prompt query key: `section_filtered_item_label`
5. Trace records `query_id=single`, `prompt_query_key=section_filtered_item_label`, section titles, item labels, marker labels, marker bboxes, item-row bboxes, trace-only reasoning bboxes for section title/filter marker/target item, sampled style metadata, and layout geometry.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
