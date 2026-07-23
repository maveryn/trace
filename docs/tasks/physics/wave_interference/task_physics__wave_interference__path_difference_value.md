# `task_physics__wave_interference__path_difference_value`

## Summary
- Domain: `physics`
- Scene id: `wave_interference`
- Implementation scene: `wave_interference`
- Implementation source: `src/trace_tasks/tasks/physics/wave_interference/path_difference_value.py`

## Program Contract

Program: `integer(abs(distance(source_s1, point_p) - distance(source_s2, point_p)) / lambda_half_step); scene=wave_interference; scope=path_difference_value`

Candidate set: the visible wave sources, wavefront spacing, candidate points, and guided source-to-point paths inside the `path_difference_value` objective scope.
Operands: `lambda_half_step` (semantic_role, allowed `visible_lambda_over_two_unit`, source `program_schema_concrete`); `point_p` (semantic_role, allowed `visible_point_P`, source `program_schema_concrete`); `source_s1` (semantic_role, allowed `visible_source_S1`, source `program_schema_concrete`); `source_s2` (semantic_role, allowed `visible_source_S2`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is an exact integer produced by the symbolic physics construction.
Annotation witnesses: `segment_set` witnesses from the finalized render. Annotation is the unordered pair of final-image pixel line segments from S1 to P and from S2 to P. Segment endpoint order is not semantic. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `abs(distance(source_s1, point_p) - distance(source_s2, point_p)) / lambda_half_step; scene=wave_interference; scope=path_difference_value` |

## Program Metadata
- Program signatures: `physics.wave_path_difference_value`
- Base program contract: `integer(abs(distance(source_s1, point_p) - distance(source_s2, point_p)) / lambda_half_step); scene=wave_interference; scope=path_difference_value`
- Parameter axes: `fixed_query`
- Arguments:
  - `lambda_half_step`: semantic_role; allowed `visible_lambda_over_two_unit`; source `program_schema_concrete`
  - `point_p`: semantic_role; allowed `visible_point_P`; source `program_schema_concrete`
  - `source_s1`: semantic_role; allowed `visible_source_S1`; source `program_schema_concrete`
  - `source_s2`: semantic_role; allowed `visible_source_S2`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`
- Internal trace query branch: `path_difference_value`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is an exact integer produced by the symbolic physics construction.

## Annotation Contract
- Annotation schema: `segment_set`
- Generator `annotation_gt.type`: `segment_set`
- Annotation is the unordered pair of final-image pixel line segments from S1 to P and from S2 to P. Segment endpoint order is not semantic.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/wave_interference/physics_wave_interference_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
