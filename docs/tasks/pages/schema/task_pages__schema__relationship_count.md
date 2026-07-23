# `task_pages__schema__relationship_count`

## Identity
- Domain: `pages`
- Scene id: `schema`
- Task id: `task_pages__schema__relationship_count`

## Program Contract
1. Program schema: `schema_relationship_count(relationship_lines) -> relationship_count; scene=schema; scope=relationship_count`
2. Scene: `schema`
3. Scope: one rendered database schema diagram with table boxes, field rows, relationship lines, labels, and cardinality markers.
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `segment_set`
7. Annotation roles: unordered visible relationship-line segments for every counted relationship.
8. Query arguments: fixed total relationship-line count; old source branch is recorded as prompt/source metadata.
9. Render arguments: table labels, field labels, key badges, relationship layout, scene variant, style variant, render dimensions, and post-render noise.

Each annotation segment is the visible endpoint-to-endpoint line witness for one
counted relationship. Relationship labels and cardinality markers are context,
not separate count witnesses.

## Reasoning Operations

Families: `counting`, `topology`
