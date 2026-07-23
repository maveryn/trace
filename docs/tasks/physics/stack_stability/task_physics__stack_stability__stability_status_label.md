# `task_physics__stack_stability__stability_status_label`

## Summary
- Domain: `physics`
- Scene id: `stack_stability`
- Implementation scene: `stack_stability`
- Implementation source: `src/trace_tasks/tasks/physics/stack_stability/stability_status_label.py`

## Task Contract
Chooses the labeled brick stack whose center-of-mass projection has the queried stability status relative to its support base.

## Query Branches

| Query id | Program schema |
| --- | --- |
| `stable_stack_label` | `option_letter(select(brick_stacks, center_of_mass_projection_inside_support_base=true)); scene=stack_stability; scope=stability_status_label` |
| `tipping_stack_label` | `option_letter(select(brick_stacks, center_of_mass_projection_inside_support_base=false)); scene=stack_stability; scope=stability_status_label` |

## Program Contract

Program: `option_letter(select(brick_stacks, center_of_mass_projection_inside_support_base=status_predicate)); scene=stack_stability; scope=stability_status_label`

Candidate set: the visible candidate stack panels, brick stacks, center-of-mass markers, projection lines, and support brackets inside the `stability_status_label` objective scope.
Operands: `brick_stacks` (visual_candidate_set, allowed `six_labeled_equal_density_brick_stacks`, source `program_schema_concrete`); `center_of_mass_projection` (semantic_role, allowed `red_com_marker_with_vertical_projection`, source `program_schema_concrete`); `support_base` (semantic_role, allowed `bottom_brick_support_footprint_bracket`, source `program_schema_concrete`); `status_predicate` (query_operand, allowed `stable|tipping`, source `query_id`); active `query_id` branch when present.
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the label of the unique stack matching the queried stability status.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation must mark a single box around the selected stack region in the final rendered image, including the brick stack, red center-of-mass marker, dashed projection line, and support-footprint bracket. Annotation must not mark option letters, unrelated candidate stacks, decorative cell panels, background grid, prompt text, or answer labels.
Query ids: `stable_stack_label`, `tipping_stack_label`.

## Reasoning Operations

Families: `aggregation`, `spatial_relations`, `formula_evaluation`

## Program Metadata
- Program signatures: `physics.stability_status_label`
- Base program contract: `option_letter(select(brick_stacks, center_of_mass_projection_inside_support_base=status_predicate)); scene=stack_stability; scope=stability_status_label`
- Parameter axes: `query_id`, `correct_option_letter`, `tip_direction`, `stack_offset_profile`
- Arguments:
  - `brick_stacks`: visual_candidate_set; allowed `six_labeled_equal_density_brick_stacks`; source `program_schema_concrete`
  - `center_of_mass_projection`: semantic_role; allowed `red_com_marker_with_vertical_projection`; source `program_schema_concrete`
  - `support_base`: semantic_role; allowed `bottom_brick_support_footprint_bracket`; source `program_schema_concrete`
  - `status_predicate`: query_operand; allowed `stable|tipping`; source `query_id`
- Argument metadata status: `curated`
- Supported query ids: `stable_stack_label`, `tipping_stack_label`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the label of the unique stack matching the queried stability status.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation must mark a single box around the selected stack region in the final rendered image, including the brick stack, red center-of-mass marker, dashed projection line, and support-footprint bracket. Annotation must not mark option letters, unrelated candidate stacks, decorative cell panels, background grid, prompt text, or answer labels.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, stack offset profiles, brick colors, option mapping, COM locations, support bounds, and verifier payloads must be explicit in the instance trace.
- Brick color, option label, row count, and tip direction must stay non-semantic; only the COM projection relative to the support footprint determines the answer.
