# `task_pages__process_flow__lane_filtered_handoff_count`

## Identity
1. Domain: `pages`
2. Scene id: `process_flow`
3. Source scene: `process_flow`
4. Task id: `task_pages__process_flow__lane_filtered_handoff_count`

## Contract
1. Objective: count visible cross-lane handoff arrows for one named lane.
2. Public task contract: `lane_filtered_handoff_count`
3. Supported `query_id` values: `lane_outgoing_handoff_count`, `lane_involved_handoff_count`
4. Answer type: `integer`
5. Annotation schema: `segment_set`
6. Annotation witness: one segment per counted handoff arrow, using the rendered source-to-destination arrow endpoints.
7. Query argument axes: lane relation mode, target lane, process context, lane count, node count, layout variant, and style variant.

## Program Contract
- `process_flow_lane_filtered_handoff_count(lane_name, relation_mode); output=integer_value; annotation=segment_set(matching_handoff_arrows); scene=process_flow; scope=one process-flow diagram`

## Reasoning Operations

Families: `counting`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_process_flow_v1`
2. Scene key: `process_flow_diagram`
3. Task key: `process_flow_diagram_query`
4. Prompt query keys: `lane_outgoing_handoff_count`, `lane_involved_handoff_count`
5. Trace records lane membership for each node, all edge specs, selected lane filter payload, final arrow segments, layout geometry, and prompt metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized process-flow render metadata.
