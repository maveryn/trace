# `task_physics__manometer__pressure_difference_value`

## Summary
- Domain: `physics`
- Scene id: `manometer`
- Implementation scene: `manometer`
- Implementation source: `src/trace_tasks/tasks/physics/manometer/pressure_difference_value.py`

## Program Contract

Program: `integer(abs(height_cm * kpa_per_cm)); scene=manometer; scope=pressure_difference_value`

Candidate set: the visible manometer columns, height difference marker, density label, and pressure side labels inside the `pressure_difference_value` objective scope.
Operands: `point_a` (semantic_role, allowed `visible_left_pressure_point_label`, source `program_schema_concrete`); `point_b` (semantic_role, allowed `visible_right_pressure_point_label`, source `program_schema_concrete`); `height_difference` (query_operand, allowed `integer_centimeter_liquid_level_difference`, source `program_schema_concrete`); `rho_g_conversion` (query_operand, allowed `integer_kpa_per_cm_conversion_label`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the absolute pressure difference in `kPa`.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation keys: `height_difference`, `fluid_density_label` Annotation must mark the visible height-difference marker and the visible conversion label needed to compute the pressure difference. It must not mark the A/B pressure point labels, decorative glass, background grid lines, or derived answer text.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(abs_pressure_difference(point_a, point_b, height_difference, rho_g_conversion)); scene=manometer; scope=pressure_difference_value` |

## Program Metadata
- Program signatures: `physics.manometer_pressure_difference`
- Base program contract: `integer(abs(height_cm * kpa_per_cm)); scene=manometer; scope=pressure_difference_value`
- Parameter axes: `height_cm`, `kpa_per_cm`, `higher_pressure_side`
- Arguments:
  - `point_a`: semantic_role; allowed `visible_left_pressure_point_label`; source `program_schema_concrete`
  - `point_b`: semantic_role; allowed `visible_right_pressure_point_label`; source `program_schema_concrete`
  - `height_difference`: query_operand; allowed `integer_centimeter_liquid_level_difference`; source `program_schema_concrete`
  - `rho_g_conversion`: query_operand; allowed `integer_kpa_per_cm_conversion_label`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the absolute pressure difference in `kPa`.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys: `height_difference`, `fluid_density_label`
- Annotation must mark the visible height-difference marker and the visible conversion label needed to compute the pressure difference. It must not mark the A/B pressure point labels, decorative glass, background grid lines, or derived answer text.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics manometer prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, height difference, conversion factor, side orientation, fluid color, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep the height marker and conversion label readable and avoid multi-fluid stacks in the current task contract.
