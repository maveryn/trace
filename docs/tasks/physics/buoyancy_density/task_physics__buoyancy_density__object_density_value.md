# `task_physics__buoyancy_density__object_density_value`

## Summary
- Domain: `physics`
- Scene id: `buoyancy_density`
- Implementation scene: `buoyancy_density`
- Implementation source: `src/trace_tasks/tasks/physics/buoyancy_density/object_density_value.py`

## Task Contract
Computes the density of a floating object from the visible submerged fraction and the shown liquid density.

## Program Contract

Program: `number(liquid_density * submerged_fraction(floating_object, waterline, equal_part_marker)); scene=buoyancy_density; scope=object_density_value`

Candidate set: the visible floating object, liquid region, level markers, and density labels inside the `object_density_value` objective scope.
Operands: `floating_object` (semantic_role, allowed `visible_divided_floating_body`, source `program_schema_concrete`); `waterline` (semantic_role, allowed `visible_liquid_surface_crossing_object`, source `program_schema_concrete`); `equal_part_marker` (semantic_role, allowed `visible_equal_division_marker`, source `program_schema_concrete`); `liquid_density` (query_operand, allowed `visible_density_label_g_cm3`, source `program_schema_concrete`).
Operation: evaluate `number` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `number` schema; Answer precision: `one_decimal`
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation marks one bounding box around the floating object. The waterline, density label, and fraction marker remain visible context in the image and trace metadata, but they are not public annotation witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `liquid_density * submerged_fraction(floating_object, waterline, equal_part_marker); scene=buoyancy_density; scope=object_density_value` |

## Program Metadata
- Program signatures: `physics.buoyancy_density.object_density`
- Base program contract: `liquid_density * submerged_fraction(floating_object, waterline, equal_part_marker); scene=buoyancy_density; scope=object_density_value`
- Parameter axes: `scene_variant`, `object_shape` (`block`, `rounded_block`), `submerged_fraction`, `liquid_density`, `target_answer`
- Arguments:
  - `floating_object`: semantic_role; allowed `visible_divided_floating_body`; source `program_schema_concrete`
  - `waterline`: semantic_role; allowed `visible_liquid_surface_crossing_object`; source `program_schema_concrete`
  - `equal_part_marker`: semantic_role; allowed `visible_equal_division_marker`; source `program_schema_concrete`
  - `liquid_density`: query_operand; allowed `visible_density_label_g_cm3`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `number`
- Answer precision: `one_decimal`
- Generator `answer_gt.type`: `number`
- The answer value is the object density in `g/cm^3` rounded to one decimal place.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation marks one bounding box around the floating object.
- The waterline, density label, and fraction marker remain visible context in the image and trace metadata, but they are not public annotation witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, scene variant, object shape, submerged fraction, liquid density, target answer, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep the liquid surface, floating object divisions, fraction marker, and liquid-density label readable.
- Object color, liquid color, object shape, and scene variant are non-semantic visual variation axes and must not be tied to the answer.
