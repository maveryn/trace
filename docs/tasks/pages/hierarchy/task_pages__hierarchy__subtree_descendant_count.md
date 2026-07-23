# `task_pages__hierarchy__subtree_descendant_count`

## Identity
1. Domain: `pages`
2. Scene id: `hierarchy`
3. Source path: `src/trace_tasks/tasks/pages/hierarchy/subtree_descendant_count.py`
4. Task id: `task_pages__hierarchy__subtree_descendant_count`

## Program Contract
1. Program schema: `org_chart.total_reports_under_manager(manager=resolved_label); scene=hierarchy; scope=one organization chart`
2. Contract: count every visible employee below the named manager through all reporting levels, excluding the named manager.
3. Public query contract: fixed `single` query; the named manager is a sampled operand recorded in trace metadata.
4. Answer schema: `integer`
5. Annotation schema: `bbox_set` with one box for every counted employee under the named manager.
6. Supported `query_id`: `single`
7. Prompt query key: `subtree_descendant_count`
8. scalar_annotation_checked=true

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_hierarchy_v1`
2. Scene key: `hierarchy_diagram`
3. Task key: `org_chart_query`
4. Prompt query key: `subtree_descendant_count`
5. Trace records the public `query_id`, prompt query key, sampled manager label, org-chart employee boxes, reporting lines, answer count, and counted employee ids.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized hierarchy render metadata.

## Rendering Notes
1. The hierarchy scene is rendered as an organization chart with the CEO at the top and reporting lines between employees.
2. User-facing prompts should avoid graph/tree terms such as node, leaf, subtree, path, and hop.
