# `task_pages__instruction_panel__shared_control_for_step_set_label`

## Identity
1. Domain: `pages`
2. Scene id: `instruction_panel`
3. Source scene: `instruction_panel`
4. Task id: `task_pages__instruction_panel__shared_control_for_step_set_label`

## Program Contract
1. Program schema: `instruction_panel_shared_control_for_step_set_label(step_number_set) -> control_label; scene=instruction_panel; scope=shared_control_for_step_set_label`
2. Scene: `instruction_panel`
3. Scope: one rendered instruction-panel page with numbered steps and visible control chips.
4. Supported `query_id`: `single`
5. Answer schema: `string`
6. Annotation schema: `bbox_set`
7. Annotation witnesses: the matching shared control-chip boxes in the referenced steps.
8. Query arguments: referenced step-number set and the uniquely shared control label.
9. Render arguments: step count, control count, controls per step, step-set size, target step set, target control, and scene layout variant.

## Reasoning Operations

Families: `matching`

## Prompt + Trace
1. Prompt bundle: `pages_instruction_panel_v1`
2. Scene key: `instruction_panel`
3. Task key: `instruction_panel_query`
4. Prompt query key: `shared_control_for_step_set_label`
5. Trace records `query_id=single`, `prompt_query_key=shared_control_for_step_set_label`, numbered steps, control labels, final chip bboxes, step-number badge bboxes, sampled style metadata, and layout geometry. Public annotation is an unordered homogeneous box set of the matching shared control chips; referenced step-number boxes remain in projected trace diagnostics.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
