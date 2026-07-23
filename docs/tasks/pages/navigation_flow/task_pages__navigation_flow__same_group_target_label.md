# `task_pages__navigation_flow__same_group_target_label`

## Identity
1. Domain: `pages`
2. Scene id: `navigation_flow`
3. Source scene: `navigation_flow`
4. Task id: `task_pages__navigation_flow__same_group_target_label`

## Contract
1. Objective: identify the other lettered control in the same visible group as a named reference candidate letter.
2. Public task contract: `same_group_target_label`
3. Supported `query_id` values: `single`
4. Answer type: `option_letter`
5. Annotation schema: `bbox`
6. Annotation witness: the selected target control bounding box.
7. Query argument axes: navigation surface, reference group, reference candidate letter, target label, scene variant, and information-scene treatment.

## Program Contract
- `same_group_target(reference_candidate_letter); output=option_letter; annotation=bbox(target_control); scene=navigation_flow; scope=one desktop application navigation screen`

## Reasoning Operations

Families: `topology`

## Prompt + Trace
1. Prompt bundle: `pages_navigation_flow_v1`
2. Scene key: `navigation_flow`
3. Task key: `navigation_path_query`
4. Prompt query key: `same_group_target_label`
5. Runtime `query_id` is `single`; selected surface and reference-control metadata are recorded in trace metadata.
6. Public annotation is the selected target control bbox. The reference control bbox is recorded in trace metadata as `reference_control_bbox_px`.
7. Generation forces two lettered controls per visible group for this task so the same-group target is unique by construction.
8. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
