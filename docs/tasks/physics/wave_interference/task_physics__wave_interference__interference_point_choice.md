# `task_physics__wave_interference__interference_point_choice`

## Summary
- Domain: `physics`
- Scene id: `wave_interference`
- Implementation scene: `wave_interference`
- Implementation source: `src/trace_tasks/tasks/physics/wave_interference/interference_point_choice.py`

## Program Contract

Program: `option_letter(select(candidate_points, interference_condition(path_difference_parity, phase_relation)=target_condition)); scene=wave_interference; scope=interference_point_choice`

Candidate set: the visible wave sources, wavefront spacing, candidate points, and guided source-to-point paths inside the `interference_point_choice` objective scope.
Operands: `candidate_points` (semantic_role, allowed `visible_candidate_points`, source `program_schema_concrete`); `path_difference_parity` (semantic_role, allowed `candidate_path_difference_parity`, source `program_schema_concrete`); `phase_relation` (semantic_role, allowed `in_phase`, `opposite_phase`, source `program_schema_concrete`); `target_condition` (query_predicate, allowed `constructive`, `destructive`, source `query_id`); active `query_id` branch when present.
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter.
Annotation witnesses: `point` witnesses from the finalized render. Annotation is one final-image pixel point at the center of the selected candidate point. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `constructive_interference_point_choice`, `destructive_interference_point_choice`.

## Reasoning Operations

Families: `spatial_relations`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `constructive_interference_point_choice` | `option_letter(select(candidate_points, interference_condition(path_difference_parity, phase_relation)=constructive)); scene=wave_interference; scope=interference_point_choice` |
| `destructive_interference_point_choice` | `option_letter(select(candidate_points, interference_condition(path_difference_parity, phase_relation)=destructive)); scene=wave_interference; scope=interference_point_choice` |

## Program Metadata
- Program signatures: `physics.wave_interference_condition_choice`
- Base program contract: `option_letter(select(candidate_points, interference_condition(path_difference_parity, phase_relation)=target_condition)); scene=wave_interference; scope=interference_point_choice`
- Parameter axes: `query_id`, `phase_relation`
- Arguments:
  - `candidate_points`: semantic_role; allowed `visible_candidate_points`; source `program_schema_concrete`
  - `path_difference_parity`: semantic_role; allowed `candidate_path_difference_parity`; source `program_schema_concrete`
  - `phase_relation`: semantic_role; allowed `in_phase`, `opposite_phase`; source `program_schema_concrete`
  - `target_condition`: query_predicate; allowed `constructive`, `destructive`; source `query_id`
- Argument metadata status: `curated`
- Supported query ids: `constructive_interference_point_choice`, `destructive_interference_point_choice`
- Internal trace query branch: `interference_point_choice`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter.

## Annotation Contract
- Annotation schema: `point`
- Generator `annotation_gt.type`: `point`
- Annotation is one final-image pixel point at the center of the selected candidate point.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/wave_interference/physics_wave_interference_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
