# `task_pages__schema__relationship_endpoint_label`

## Identity
- Domain: `pages`
- Scene id: `schema`
- Task id: `task_pages__schema__relationship_endpoint_label`

## Program Contract
1. Program schema: `schema_relationship_endpoint_label(source_table, relationship_label) -> target_table_label; scene=schema; scope=relationship_endpoint_label`
2. Scene: `schema`
3. Scope: one rendered database schema diagram with table boxes, field rows, relationship lines, labels, and cardinality markers.
4. Supported `query_id` values: `single`
5. Answer schema: `string`
6. Answer support: exact visible table labels.
7. Annotation schema: `bbox`
8. Annotation roles: one box around the target table named by the answer.
9. Query arguments: fixed labeled-relationship endpoint lookup from one named source table; old source branch is recorded as prompt/source metadata.
10. Render arguments: table labels, field labels, key badges, relationship layout, scene variant, style variant, render dimensions, and post-render noise.

Annotation marks only the target table named by the answer. The source table and
relationship label are prompt/context witnesses retained in the trace payload,
not public annotation. The selected relationship is unique for the sampled
source table and relationship label.

## Reasoning Operations

Families: `topology`
