# `task_pages__schema__field_role_count`

## Identity
- Domain: `pages`
- Scene id: `schema`
- Task id: `task_pages__schema__field_role_count`

## Program Contract
1. Program schema: `schema_field_role_count(table, field_role) -> field_count; scene=schema; scope=field_role_count`
2. Scene: `schema`
3. Scope: one rendered database schema diagram with table boxes, field rows, relationship lines, labels, and cardinality markers.
4. Supported `query_id` values: `all_field_count`, `attribute_field_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Annotation roles: unordered field-row boxes for every counted row in the named table.
8. Query arguments: `field_role=all` for all rows or `field_role=attribute` for rows without `PK` or `FK` badges.
9. Render arguments: table labels, field labels, key badges, relationship layout, scene variant, style variant, render dimensions, and post-render noise.

The `all_field_count` branch counts every visible field row in the named table.
The `attribute_field_count` branch counts only rows without `PK` or `FK` badges.
Annotation boxes mark the counted field rows only.

## Reasoning Operations

Families: `filtering`, `counting`
