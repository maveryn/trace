# `task_physics__orbital_motion__orbital_speed_extremum_label`

## Summary
- Domain: `physics`
- Scene id: `orbital_motion`
- Implementation scene: `orbital_motion`
- Implementation source: `src/trace_tasks/tasks/physics/orbital_motion/orbital_speed_extremum_label.py`

## Task Contract
Selects the labeled orbital position where speed is greatest or least among four visible candidate positions, using the Sun-at-focus distance relationship.

## Program Contract

Program: `option_letter(arg_extreme(candidate_orbit_positions, distance_to_sun_focus, direction=speed_extremum)); scene=orbital_motion; scope=orbital_speed_extremum_label`

Candidate set: the visible orbit, focus candidates, planet-position candidates, and orbital labels inside the `orbital_speed_extremum_label` objective scope.
Operands: `candidate_orbit_positions` (semantic_role, allowed `visible_labeled_positions_on_orbit`, source `program_schema_concrete`); `sun_focus` (semantic_role, allowed `visible_sun_at_focus`, source `program_schema_concrete`); `speed_extremum_direction` (semantic_role, allowed `greatest`, `least`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible candidate label.
Annotation witnesses: `point` witnesses from the finalized render. Annotation marks one pixel point at the center of the selected planet-position marker. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `greatest_speed_position_label`, `least_speed_position_label`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `greatest_speed_position_label` | `option_letter(arg_extreme(candidate_orbit_positions, distance_to_sun_focus, direction=nearest)); scene=orbital_motion; scope=orbital_speed_extremum_label; query_branch=greatest_speed_position_label` |
| `least_speed_position_label` | `option_letter(arg_extreme(candidate_orbit_positions, distance_to_sun_focus, direction=farthest)); scene=orbital_motion; scope=orbital_speed_extremum_label; query_branch=least_speed_position_label` |

## Program Metadata
- Program signatures: `physics.orbital_speed_extremum_label`
- Base program contract: `option_letter(arg_extreme(candidate_orbit_positions, distance_to_sun_focus, direction=speed_extremum)); scene=orbital_motion; scope=orbital_speed_extremum_label`
- Parameter axes: `speed_extremum_direction`
- Arguments:
  - `candidate_orbit_positions`: semantic_role; allowed `visible_labeled_positions_on_orbit`; source `program_schema_concrete`
  - `sun_focus`: semantic_role; allowed `visible_sun_at_focus`; source `program_schema_concrete`
  - `speed_extremum_direction`: semantic_role; allowed `greatest`, `least`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `greatest_speed_position_label`, `least_speed_position_label`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible candidate label.

## Annotation Contract
- Annotation schema: `point`
- Generator `annotation_gt.type`: `point`
- Annotation marks one pixel point at the center of the selected planet-position marker.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics orbital-motion v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
