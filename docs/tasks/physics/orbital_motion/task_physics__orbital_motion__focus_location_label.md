# `task_physics__orbital_motion__focus_location_label`

## Summary
- Domain: `physics`
- Scene id: `orbital_motion`
- Implementation scene: `orbital_motion`
- Implementation source: `src/trace_tasks/tasks/physics/orbital_motion/focus_location_label.py`

## Task Contract
Selects the labeled point that can be the Sun at a focus of an elliptical orbit. The rendered diagram shows six labeled candidate points, including major-axis and off-axis distractors.

## Program Contract

Program: `option_letter(select(candidate_points, point_is_focus_of_ellipse)); scene=orbital_motion; scope=focus_location_label`

Candidate set: the visible orbit, focus candidates, planet-position candidates, and orbital labels inside the `focus_location_label` objective scope.
Operands: `candidate_points` (semantic_role, allowed `visible_labeled_candidate_points`, source `program_schema_concrete`); `ellipse_geometry` (semantic_role, allowed `visible_ellipse_center_and_major_axis`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible candidate label.
Annotation witnesses: `point` witnesses from the finalized render. Annotation marks one pixel point at the center of the selected focus candidate marker. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(select(candidate_points, point_is_focus_of_ellipse)); scene=orbital_motion; scope=focus_location_label` |

## Program Metadata
- Program signatures: `physics.orbital_focus_location_label`
- Base program contract: `option_letter(select(candidate_points, point_is_focus_of_ellipse)); scene=orbital_motion; scope=focus_location_label`
- Parameter axes: `fixed_query`
- Arguments:
  - `candidate_points`: semantic_role; allowed `visible_labeled_candidate_points`; source `program_schema_concrete`
  - `ellipse_geometry`: semantic_role; allowed `visible_ellipse_center_and_major_axis`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible candidate label.

## Annotation Contract
- Annotation schema: `point`
- Generator `annotation_gt.type`: `point`
- Annotation marks one pixel point at the center of the selected focus candidate marker.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics orbital-motion v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
