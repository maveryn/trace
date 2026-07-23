# `task_physics__gear_train__output_direction_label`

## Summary
- Domain: `physics`
- Scene id: `gear_train`
- Implementation scene: `gear_train`
- Implementation source: `src/trace_tasks/tasks/physics/gear_train/output_direction_label.py`

## Task Contract
Selects the one labeled gear-train panel whose marked output gear rotates in the requested direction.

## Program Contract

Program: `option_letter(unique_panel_where(propagate_adjacent_mesh_reversals(input_direction, gear_count) == target_direction)); scene=gear_train; scope=output_direction_label`

Candidate set: the visible gears, tooth-count labels, input/output markers, and candidate gear-train panels inside the `output_direction_label` objective scope.
Operands: `panel_gear_trains` (visual_candidate_set, allowed `four_labeled_panels_A_D_each_with_input_rotation_arrow_and_marked_output_gear`, source `program_schema_concrete`); `target_direction` (query_operand, allowed `clockwise_or_counterclockwise`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the one panel letter, `A`, `B`, `C`, or `D`, whose marked output gear rotates in the requested direction.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation marks the selected labeled gear-train panel. Scalar annotation checked: `true`; exactly one option panel is selected by construction.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(unique_panel_where(direction(propagate_adjacent_mesh_reversals(input_gear_rotation, gear_count), target=marked_output_gear) == target_direction)); scene=gear_train; scope=output_direction_label` |

## Program Metadata
- Program signatures: `physics.gear_train_output_direction_choice`
- Base program contract: `option_letter(unique_panel_where(propagate_adjacent_mesh_reversals(input_direction, gear_count) == target_direction)); scene=gear_train; scope=output_direction_label`
- Parameter axes: `target_direction`, `correct_option_letter`, `scene_variant`, `gear_count`, `input_direction`, `gear_radii`, `gear_layout`
- Arguments:
  - `panel_gear_trains`: visual_candidate_set; allowed `four_labeled_panels_A_D_each_with_input_rotation_arrow_and_marked_output_gear`; source `program_schema_concrete`
  - `target_direction`: query_operand; allowed `clockwise_or_counterclockwise`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the one panel letter, `A`, `B`, `C`, or `D`, whose marked output gear rotates in the requested direction.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation marks the selected labeled gear-train panel.
- Scalar annotation checked: `true`; exactly one option panel is selected by construction.
- Annotation must not mark a solved output arrow, decorative panel/background elements, or prompt text.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics gear-train v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, target direction, correct option letter, panel gear counts, gear radii, input directions, layout variants, colors, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep adjacent gear contacts visually unambiguous and must not draw solved output-direction arrows.
