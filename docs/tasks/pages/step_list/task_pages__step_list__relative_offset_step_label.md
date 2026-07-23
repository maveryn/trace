# `task_pages__step_list__relative_offset_step_label`

## Identity
1. Domain: `pages`
2. Scene id: `step_list`
3. Source scene: `step_list`
4. Task id: `task_pages__step_list__relative_offset_step_label`

## Program Contract
1. Program schema: `step_list_relative_offset_step_label(source_step_title, offset_relation={before,after}, offset_count) -> target_step_title; scene=step_list; scope=relative_offset_step_label`
2. Scene: `step_list`
3. Scope: one rendered numbered workflow step list with visible step numbers, labeled Title and Detail fields, and non-answer Owner/Status/Due/Tag fields.
4. Supported `query_id`: `offset_after_named_step`, `offset_before_named_step`
5. Answer schema: `string`
6. Annotation schema: `bbox`
7. Annotation witness: one box around the answer step title.
8. Query arguments: visible source step title, before/after relation, and integer step offset.
9. Render arguments: step count, scene layout variant, source step index, offset count, compact card fields, visual style, and post-render noise.

## Reasoning Operations

Families: `ranking`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_step_list_v1`
2. Scene key: `step_list`
3. Task key: `step_lookup_query`
4. Prompt query keys: `offset_after_named_step`, `offset_before_named_step`
5. Trace records selected query id, prompt query key, source and target step records, offset relation/count, title bboxes, sampled layout metadata, and scalar projected annotation.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
