# `task_physics__fluid_flow__continuity_speed_value`

## Summary
- Domain: `physics`
- Scene id: `fluid_flow`
- Implementation scene: `fluid_flow`
- Implementation source: `src/trace_tasks/tasks/physics/fluid_flow/continuity_speed_value.py`

## Task Contract
Computes a missing steady-flow speed from a two-station pipe/nozzle diagram using incompressible-flow continuity.

## Program Contract

Program: `integer(solve(A1 * v1 = A2 * v2, unknown_speed)); scene=fluid_flow; scope=continuity_speed_value`

Candidate set: the visible pipe sections, area labels, speed labels, and marked missing readout inside the `continuity_speed_value` objective scope.
Operands: `station_1` (query_operand, allowed `visible_station_1_area_and_speed_labels`, source `program_schema_concrete`); `station_2` (query_operand, allowed `visible_station_2_area_and_speed_labels`, source `program_schema_concrete`); `missing_speed_label` (queried_slot, allowed `visible_v1_or_v2_missing_speed_label`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the missing positive integer flow speed in `m/s`.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation is the final-image pixel bbox around the visible missing speed label, such as `v1 = ?` or `v2 = ?`. Annotation must mark the queried missing-value slot, not the full station regions, flow path, decorative background grid lines, panel framing, or derived answer text.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(solve(A1 * v1 = A2 * v2, unknown_speed)); scene=fluid_flow; scope=continuity_speed_value` |

## Program Metadata
- Program signatures: `physics.fluid_flow_continuity_speed`
- Base program contract: `integer(solve_continuity_speed(area_1_cm2, speed_1_m_s, area_2_cm2, speed_2_m_s, unknown_speed_slot)); scene=fluid_flow; scope=continuity_speed_value`
- Parameter axes: `orientation`, `missing_speed_station`, `area_1_cm2`, `area_2_cm2`, `speed_1_m_s`, `speed_2_m_s`
- Arguments:
  - `station_1`: query_operand; allowed `visible_station_1_area_and_speed_labels`; source `program_schema_concrete`
  - `station_2`: query_operand; allowed `visible_station_2_area_and_speed_labels`; source `program_schema_concrete`
  - `missing_speed_label`: queried_slot; allowed `visible_v1_or_v2_missing_speed_label`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the missing positive integer flow speed in `m/s`.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation is the final-image pixel bbox around the visible missing speed label, such as `v1 = ?` or `v2 = ?`.
- Annotation must mark the queried missing-value slot, not the full station regions, flow path, decorative background grid lines, panel framing, or derived answer text.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.
- Scalar annotation checked: `true`; a single bbox is used because each instance has exactly one missing speed label.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/fluid_flow/physics_fluid_flow_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, station areas, station speeds, missing station, orientation, fluid color, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep both station labels readable, show one clear missing speed label, and use explicit area labels rather than diameter labels in the current task contract.
