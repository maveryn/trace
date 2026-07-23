# `task_pages__step_list__between_named_steps_count`

## Identity
1. Domain: `pages`
2. Scene id: `step_list`
3. Source scene: `step_list`
4. Task id: `task_pages__step_list__between_named_steps_count`

## Program Contract
1. Program schema: `step_list_between_named_steps_count(first_step_title, second_step_title) -> count_between; scene=step_list; scope=between_named_steps_count`
2. Scene: `step_list`
3. Scope: one rendered numbered workflow step list with visible step numbers, labeled Title and Detail fields, and non-answer Owner/Status/Due/Tag fields.
4. Supported `query_id`: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_map`
7. Annotation witness: boxes around the two named boundary step titles, keyed as `first_named_title` and `second_named_title`.
8. Query arguments: two visible boundary step titles.
9. Render arguments: step count, scene layout variant, boundary step indices, compact card fields, visual style, and post-render noise.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_step_list_v1`
2. Scene key: `step_list`
3. Task key: `step_lookup_query`
4. Prompt query key: `between_named_steps_count`
5. Trace records `query_id=single`, prompt query key, boundary step records, requested between-count operand, title bboxes, sampled layout metadata, and projected bbox-map annotation.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
