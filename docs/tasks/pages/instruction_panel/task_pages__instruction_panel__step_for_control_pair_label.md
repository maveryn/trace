# `task_pages__instruction_panel__step_for_control_pair_label`

## Identity
1. Domain: `pages`
2. Scene id: `instruction_panel`
3. Source scene: `instruction_panel`
4. Task id: `task_pages__instruction_panel__step_for_control_pair_label`

## Program Contract
1. Program schema: `instruction_panel_step_for_control_pair_label(control_label_a, control_label_b) -> step_number; scene=instruction_panel; scope=step_for_control_pair_label`
2. Scene: `instruction_panel`
3. Scope: one rendered instruction-panel page with numbered steps and visible control chips.
4. Supported `query_id`: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Annotation witnesses: the two matching control-chip boxes and the answer step-number badge box.
8. Query arguments: referenced control-label pair and the unique step containing both controls.
9. Render arguments: step count, control count, controls per step, target control pair, target step, and scene layout variant.

## Reasoning Operations

Families: `matching`

## Prompt + Trace
1. Prompt bundle: `pages_instruction_panel_v1`
2. Scene key: `instruction_panel`
3. Task key: `instruction_panel_query`
4. Prompt query key: `step_for_control_pair_label`
5. Trace records `query_id=single`, `prompt_query_key=step_for_control_pair_label`, numbered steps, control labels, final chip bboxes, step-number badge bboxes, sampled style metadata, and layout geometry. Public annotation is an unordered homogeneous box set; role-keyed boxes remain in projected trace diagnostics.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
