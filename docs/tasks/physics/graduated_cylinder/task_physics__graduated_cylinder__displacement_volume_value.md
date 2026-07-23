# `task_physics__graduated_cylinder__displacement_volume_value`

## Summary
- Domain: `physics`
- Scene id: `graduated_cylinder`
- Implementation scene: `graduated_cylinder`
- Implementation source: `src/trace_tasks/tasks/physics/graduated_cylinder/displacement_volume_value.py`

## Task Contract
Computes displaced volume from before/after graduated-cylinder readings.

## Program Contract

Program: `integer(read_scale_value(after_meniscus, graduated_scale) - read_scale_value(before_meniscus, graduated_scale)); scene=graduated_cylinder; scope=displacement_volume_value`

Candidate set: the visible cylinder readouts, liquid levels, tick marks, numeric labels, and unit labels inside the `displacement_volume_value` objective scope.
Operands: `before_meniscus` (semantic_role, allowed `visible_before_liquid_level`, source `program_schema_concrete`); `after_meniscus` (semantic_role, allowed `visible_after_liquid_level`, source `program_schema_concrete`); `graduated_scale` (semantic_role, allowed `matched_visible_tick_scales_with_mL_units`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_value` schema; The answer value is the exact integer mL increase from the before reading to the after reading.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation is keyed because before/after witness roles are distinct; keys are `before_cylinder` and `after_cylinder`. Each annotation box marks the corresponding graduated-cylinder readout, including the cylinder body, liquid level, tick marks, numeric scale labels, and mL unit.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `read_scale_value(after_meniscus, graduated_scale) - read_scale_value(before_meniscus, graduated_scale); scene=graduated_cylinder; scope=displacement_volume_value; query_branch=before_after_displacement_volume` |

## Program Metadata
- Program signatures: `physics.graduated_cylinder_displacement_volume`
- Base program contract: `read_scale_value(after_meniscus, graduated_scale) - read_scale_value(before_meniscus, graduated_scale); scene=graduated_cylinder; scope=displacement_volume_value`
- Parameter axes: `fixed_query`
- Arguments:
  - `before_meniscus`: semantic_role; allowed `visible_before_liquid_level`; source `program_schema_concrete`
  - `after_meniscus`: semantic_role; allowed `visible_after_liquid_level`; source `program_schema_concrete`
  - `graduated_scale`: semantic_role; allowed `matched_visible_tick_scales_with_mL_units`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer_value`
- Generator `answer_gt.type`: `integer`
- The answer value is the exact integer mL increase from the before reading to the after reading.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation is keyed because before/after witness roles are distinct; keys are `before_cylinder` and `after_cylinder`.
- Each annotation box marks the corresponding graduated-cylinder readout, including the cylinder body, liquid level, tick marks, numeric scale labels, and mL unit.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.
- Scalar annotation checked: `true`

## Prompt And Trace Requirements
- Prompt text must come from the physics graduated-cylinder v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
