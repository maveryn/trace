# `task_physics__free_body_forces__net_force_direction_choice`

## Summary
- Domain: `physics`
- Scene id: `free_body_forces`
- Implementation scene: `free_body_forces`
- Implementation source: `src/trace_tasks/tasks/physics/free_body_forces/net_force_direction_choice.py`

## Task Contract
Selects the candidate arrow showing the net-force direction for one object from visible applied-force arrows and magnitude labels.

## Program Contract

Program: `option_letter(direction(sum(applied_force_vectors))); scene=free_body_forces; scope=net_force_direction_choice`

Candidate set: the visible force diagram, applied force arrows, magnitude labels, and candidate result arrows inside the `net_force_direction_choice` objective scope.
Operands: `applied_force_vectors` (visual_input_set, allowed `visible_cardinal_force_arrows_with_magnitude_labels`, source `program_schema_concrete`); `candidate_net_force_arrows` (visual_candidate_set, allowed `eight_labeled_direction_arrows_A_H`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the visible option letter whose candidate arrow matches the direction of the summed applied-force vector.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation keys: `force_diagram`: box around the object, applied force arrows, and magnitude labels.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(direction(sum(applied_force_vectors))); scene=free_body_forces; scope=net_force_direction_choice` |

## Program Metadata
- Program signatures: `physics.net_force_direction_choice`
- Base program contract: `option_letter(direction(sum(applied_force_vectors))); scene=free_body_forces; scope=net_force_direction_choice`
- Parameter axes: `fixed_query`, `scene_variant`, `net_force_direction`, `correct_option_letter`, `accent_color_name`
- Arguments:
  - `applied_force_vectors`: visual_input_set; allowed `visible_cardinal_force_arrows_with_magnitude_labels`; source `program_schema_concrete`
  - `candidate_net_force_arrows`: visual_candidate_set; allowed `eight_labeled_direction_arrows_A_H`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the visible option letter whose candidate arrow matches the direction of the summed applied-force vector.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys:
  - `force_diagram`: box around the object, applied force arrows, and magnitude labels.
  - `selected_candidate`: box around the selected candidate arrow option cell.
- Annotation must mark visual witnesses from the final rendered diagram, not decorative panel chrome or any derived resultant annotation.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.
- Scalar annotation checked: `true`; keyed boxes are used because the source-force diagram and selected candidate option have distinct roles.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/free_body_forces/physics_free_body_forces_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Prompts must ask for the net-force direction, not the object's subsequent motion direction.
- Render randomness, sampled fonts/styles, force magnitudes, option mapping, answer construction, and verifier payloads must be explicit in the instance trace.
- Option letters, colors, scene variants, and canceling-force distractors must stay non-semantic; only the summed applied-force vectors determine the answer.
