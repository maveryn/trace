# `task_pages__hierarchy__manager_most_direct_reports_label`

## Identity
1. Domain: `pages`
2. Scene id: `hierarchy`
3. Source path: `src/trace_tasks/tasks/pages/hierarchy/manager_most_direct_reports_label.py`
4. Task id: `task_pages__hierarchy__manager_most_direct_reports_label`

## Program Contract
1. Program schema: `org_chart.manager_extremum(metric=direct_reports, exclude_root=true); scene=hierarchy; scope=one organization chart`
2. Contract: compare every non-CEO manager by number of immediate direct reports, and return the unique manager with the largest direct-report count.
3. Public query contract: fixed `single` query; org-chart structure, manager labels, and visual style are sampled operands/render axes recorded in trace metadata.
4. Answer schema: `string`
5. Annotation schema: scalar `bbox` around the selected manager.
6. Supported `query_id`: `single`
7. Prompt query key: `manager_most_direct_reports_label`
8. scalar_annotation_checked=true

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_hierarchy_v1`
2. Scene key: `hierarchy_diagram`
3. Task key: `org_chart_query`
4. Prompt query key: `manager_most_direct_reports_label`
5. Trace records compared non-CEO managers, each manager's direct-report count, the selected manager id/label, and the selected manager bbox.
6. Generation guarantees a unique non-CEO manager with the largest direct-report count.

## Rendering Notes
1. The top box is labeled `CEO`; prompts explicitly exclude the CEO at the top.
2. Annotation marks the selected manager only; compared manager counts are recorded in trace metadata.
