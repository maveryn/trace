# `task_physics__piston_cylinder__boundary_work_value`

## Summary
- Domain: `physics`
- Scene id: `piston_cylinder`
- Implementation scene: `piston_cylinder`
- Implementation source: `src/trace_tasks/tasks/physics/piston_cylinder/boundary_work_value.py`

## Task Contract
Computes signed constant-pressure boundary work from visible pressure, initial volume, final volume, and process direction in a piston-cylinder diagram.

## Program Contract

Program: `integer(P_MPa * (V_final_L - V_initial_L)); scene=piston_cylinder; scope=boundary_work_value`

Candidate set: the visible initial and final piston-cylinder states, pressure readout, dimensions, and motion cue inside the `boundary_work_value` objective scope.
Operands: `piston_cylinder` (semantic_role, allowed `visible_initial_and_final_piston_cylinder_apparatus`, source `program_schema_concrete`); `pressure_readout` (query_operand, allowed `visible_constant_pressure_readout`, source `program_schema_concrete`); `initial_cylinder` (query_operand, allowed `visible_initial_cylinder_state`, source `program_schema_concrete`); `final_cylinder` (query_operand, allowed `visible_final_cylinder_state`, source `program_schema_concrete`); `process_arrow` (query_operand, allowed `visible_initial_to_final_process_arrow`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the signed boundary work in `kJ`, using work done by the gas as positive.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation keys: `pressure_readout`, `initial_cylinder`, `final_cylinder` Annotation must mark the pressure readout and the visible initial and final cylinder states. It must not mark decorative background grid lines, panel framing, or derived answer text.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(boundary_work_kj(pressure_mpa, final_volume_l - initial_volume_l)); scene=piston_cylinder; scope=boundary_work_value` |

## Program Metadata
- Program signatures: `physics.piston_cylinder_boundary_work`
- Base program contract: `integer(P_MPa * (V_final_L - V_initial_L)); scene=piston_cylinder; scope=boundary_work_value`
- Parameter axes: `pressure_mpa`, `initial_volume_l`, `final_volume_l`, `orientation`
- Arguments:
  - `piston_cylinder`: semantic_role; allowed `visible_initial_and_final_piston_cylinder_apparatus`; source `program_schema_concrete`
  - `pressure_readout`: query_operand; allowed `visible_constant_pressure_readout`; source `program_schema_concrete`
  - `initial_cylinder`: query_operand; allowed `visible_initial_cylinder_state`; source `program_schema_concrete`
  - `final_cylinder`: query_operand; allowed `visible_final_cylinder_state`; source `program_schema_concrete`
  - `process_arrow`: query_operand; allowed `visible_initial_to_final_process_arrow`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the signed boundary work in `kJ`, using work done by the gas as positive.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys: `pressure_readout`, `initial_cylinder`, `final_cylinder`
- Annotation must mark the pressure readout and the visible initial and final cylinder states. It must not mark decorative background grid lines, panel framing, or derived answer text.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics piston-cylinder v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, pressure, volumes, orientation, gas color, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep pressure and volume labels readable, show a clear initial-to-final process arrow, and avoid zero-work cases in the current support.
