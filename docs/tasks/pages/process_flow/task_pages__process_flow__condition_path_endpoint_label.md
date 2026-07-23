# `task_pages__process_flow__condition_path_endpoint_label`

## Identity
1. Domain: `pages`
2. Scene id: `process_flow`
3. Source scene: `process_flow`
4. Task id: `task_pages__process_flow__condition_path_endpoint_label`

## Contract
1. Objective: follow two visible decision-arrow labels and return the reached step label.
2. Public task contract: `condition_path_endpoint_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox_map`
6. Annotation witness: keyed boxes for `start_step`, `first_decision_label`, `second_decision_label`, and `endpoint_step`.
7. Query argument axes: selected decision-label sequence, process context, lane count, node count, layout variant, and style variant.

## Program Contract
- `process_flow_condition_path_endpoint_label(decision_label_sequence); output=visible_step_label; annotation=bbox_map(path_witnesses); scene=process_flow; scope=one process-flow diagram`

## Reasoning Operations

Families: `topology`

## Prompt + Trace
1. Prompt bundle: `pages_process_flow_v1`
2. Scene key: `process_flow_diagram`
3. Task key: `process_flow_diagram_query`
4. Prompt query key: `condition_path_endpoint_label`
5. Trace records the selected decision labels, symbolic path node ids, minimal annotation witness roles, final node and arrow-label bboxes, layout geometry, and prompt metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized process-flow render metadata.
