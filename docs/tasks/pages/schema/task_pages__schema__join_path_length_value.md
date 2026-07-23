# `task_pages__schema__join_path_length_value`

## Identity
- Domain: `pages`
- Scene id: `schema`
- Task id: `task_pages__schema__join_path_length_value`

## Program Contract
1. Program schema: `schema_join_path_length(source_table, target_table) -> relationship_path_length; scene=schema; scope=join_path_length_value`
2. Scene: `schema`
3. Scope: one rendered database schema diagram with table boxes, field rows, relationship lines, labels, and cardinality markers.
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Answer support: `1`, `2`, `3`, or `4`
7. Annotation schema: `segment_set`
8. Annotation roles: ordered relationship-line segments on the unique shortest path between the requested tables.
9. Query arguments: source table label and target table label.
10. Render arguments: table labels, field labels, key badges, relationship layout, scene variant, style variant, render dimensions, and post-render noise.

The generator constructs a unique shortest relationship path of the requested
length and adds distractor relationships that do not create a shorter or tied
shortest path between the requested endpoint tables. Annotation marks only the
relationship-line segments on the shortest path. Endpoint table boxes are trace
context, not public annotation.

## Reasoning Operations

Families: `counting`, `topology`
