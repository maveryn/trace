# `task_physics__gear_train__output_speed_value`

## Summary
- Domain: `physics`
- Scene id: `gear_train`
- Implementation scene: `gear_train`
- Implementation source: `src/trace_tasks/tasks/physics/gear_train/output_speed_value.py`

## Task Contract
Computes the marked output gear speed from a simple directly meshed gear train with visible tooth-count labels and a visible input rpm label.

## Program Contract

Program: `integer(solve_simple_gear_ratio(input_rpm, input_teeth, output_teeth)); scene=gear_train; scope=output_speed_value`

Candidate set: the visible gears, tooth-count labels, input/output markers, and candidate gear-train panels inside the `output_speed_value` objective scope.
Operands: `input_gear` (query_operand, allowed `visible_input_gear_with_tooth_count_and_input_rpm_label`, source `program_schema_concrete`); `output_gear` (output_binding, allowed `visible_marked_output_gear_with_tooth_count`, source `program_schema_concrete`); `gear_train` (semantic_role, allowed `complete_visible_direct_mesh_train`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the marked output gear's integer rotational speed in `rpm`.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation keys: `input_gear`, `output_gear` Scalar annotation checked: not scalar, because role-bound boxes mark multiple distinct gear-train witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(input_rpm * input_gear_tooth_count / output_gear_tooth_count); scene=gear_train; scope=output_speed_value` |

## Program Metadata
- Program signatures: `physics.gear_train_output_speed`
- Base program contract: `integer(solve_simple_gear_ratio(input_rpm, input_teeth, output_teeth)); scene=gear_train; scope=output_speed_value`
- Parameter axes: `scene_variant`, `gear_count`, `input_teeth`, `output_teeth`, `idler_teeth`, `input_rpm`
- Arguments:
  - `input_gear`: query_operand; allowed `visible_input_gear_with_tooth_count_and_input_rpm_label`; source `program_schema_concrete`
  - `output_gear`: output_binding; allowed `visible_marked_output_gear_with_tooth_count`; source `program_schema_concrete`
  - `gear_train`: semantic_role; allowed `complete_visible_direct_mesh_train`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the marked output gear's integer rotational speed in `rpm`.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys: `input_gear`, `output_gear`
- Scalar annotation checked: not scalar, because role-bound boxes mark multiple distinct gear-train witnesses.
- Annotation must mark the input gear with its tooth-count and input-rpm labels, and the marked output gear with its tooth-count label. It must not mark idler gears, decorative panel/background elements, or derived output speed text.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics gear-train v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, gear count, tooth counts, input rpm, layout variant, colors, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep tooth-count labels and the input rpm label readable. Idler gears may appear, but first-version semantics use only the input and output tooth counts for the speed ratio.
