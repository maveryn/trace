# `task_pages__sectioned_infographic__section_item_count`

## Identity
1. Domain: `pages`
2. Scene id: `sectioned_infographic`
3. Source scene: `sectioned_infographic`
4. Task id: `task_pages__sectioned_infographic__section_item_count`

## Program Contract
1. Program schema: `sectioned_infographic_section_item_count(section_title) -> item_count; scene=sectioned_infographic; scope=section_item_count`
2. Scene: `sectioned_infographic`
3. Scope: one rendered sectioned infographic with named sections and visible item rows.
4. Supported `query_id`: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Annotation roles: unordered boxes for all visible item rows in the requested section.
8. Query arguments: resolved target section title.
9. Render arguments: section count, per-section item-count support, scene layout variant, visual style, and post-render noise.

## Reasoning Operations

Families: `counting`

## Prompt + Trace
1. Prompt bundle: `pages_sectioned_infographic_v1`
2. Scene key: `sectioned_infographic`
3. Task key: `sectioned_infographic_query`
4. Prompt query key: `section_item_count`
5. Trace records `query_id=single`, `prompt_query_key=section_item_count`, section titles, item labels, final section bboxes, final item-row bboxes, sampled style metadata, and layout geometry.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
