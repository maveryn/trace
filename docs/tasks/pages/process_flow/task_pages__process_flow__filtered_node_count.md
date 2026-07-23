# `task_pages__process_flow__filtered_node_count`

## Identity
1. Domain: `pages`
2. Scene id: `process_flow`
3. Source scene: `process_flow`
4. Task id: `task_pages__process_flow__filtered_node_count`

## Contract
1. Objective: count process-flow step boxes matching one visible node filter.
2. Public task contract: `filtered_node_count`
3. Supported `query_id` values: `shape_node_count`, `status_node_count`, `role_node_count`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: boxes around every counted step node.
7. Query argument axes: filter family, include/exclude predicate, selected shape/status/role, process context, lane count, node count, layout variant, and style variant.

## Program Contract
- `process_flow_filtered_node_count(filter_family, predicate); output=integer_value; annotation=bbox_set(counted_step_nodes); scene=process_flow; scope=one process-flow diagram`

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_process_flow_v1`
2. Scene key: `process_flow_diagram`
3. Task key: `process_flow_diagram_query`
4. Prompt query keys: `shape_node_count`, `status_node_count`, `role_node_count`
5. Trace records the sampled process context, lanes, node specs, edge specs, selected filter payload, final node bboxes, layout geometry, and prompt metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized process-flow render metadata.
