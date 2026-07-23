# `task_pages__schema__relationship_cardinality_label`

## Identity
- Domain: `pages`
- Scene id: `schema`
- Task id: `task_pages__schema__relationship_cardinality_label`

## Program Contract
1. Program schema: `schema_relationship_cardinality_label(source_table, target_table, endpoint_markers) -> cardinality_label; scene=schema; scope=relationship_cardinality_label`
2. Scene: `schema`
3. Scope: one rendered database schema diagram with table boxes, field rows, relationship lines, labels, and cardinality markers.
4. Supported `query_id` values: `single`
5. Answer schema: `string`
6. Answer support: `one_to_many`, `optional_many`, `one_to_one`, or `many_to_many`.
7. Annotation schema: `bbox_map`
8. Annotation roles: keyed boxes for `source_cardinality_marker` and `target_cardinality_marker`.
9. Query arguments: fixed cardinality-marker lookup between two named endpoint tables; old source branch is recorded as prompt/source metadata.
10. Render arguments: table labels, field labels, key badges, relationship layout, scene variant, style variant, render dimensions, and post-render noise.

Annotation marks only the two endpoint cardinality markers used to derive the
answer. Source and target table boxes are prompt/context witnesses retained in
the trace payload, not public annotation.

## Reasoning Operations

Families: `topology`
