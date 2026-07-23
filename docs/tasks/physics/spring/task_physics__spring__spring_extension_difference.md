# `task_physics__spring__spring_extension_difference`

## Summary
- Domain: `physics`
- Scene id: `spring`
- Implementation scene: `spring`
- Implementation source: `src/trace_tasks/tasks/physics/spring/spring_extension_difference.py`

## Program Contract

Program: `abs(extension_a - extension_b); scene=spring; scope=spring_extension_difference`

Candidate set: the visible spring setups, weight blocks, extension markers, and measurement labels inside the `spring_extension_difference` objective scope.
Operands: `extension_a` (semantic_role, allowed `left_spring_extension_marker`, source `program_schema_concrete`); `extension_b` (semantic_role, allowed `right_spring_extension_marker`, source `program_schema_concrete`).
Operation: evaluate `abs` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_value` schema; The answer value is the exact integer absolute difference between the two shown extension markers.
Annotation witnesses: `bbox_set` witnesses from the finalized render. Annotation must mark the two visible spring-extension markers being compared. Extension-marker boxes are vertically padded around the marker so the annotation target is not too thin.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `abs(extension_a - extension_b); scene=spring; scope=spring_extension_difference` |

## Program Metadata
- Program signatures: `physics.extension_difference_value`
- Base program contract: `abs(extension_a - extension_b); scene=spring; scope=spring_extension_difference`
- Parameter axes: `scene_variant`, `target_answer`, `accent_color_name`
- Arguments:
  - `extension_a`: semantic_role; allowed `left_spring_extension_marker`; source `program_schema_concrete`
  - `extension_b`: semantic_role; allowed `right_spring_extension_marker`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`
- Internal trace query branch: `extension_difference`

## Answer Contract
- Answer schema: `integer_value`
- Generator `answer_gt.type`: `integer`
- The answer value is the exact integer absolute difference between the two shown extension markers.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation must mark the two visible spring-extension markers being compared. Extension-marker boxes are vertically padded around the marker so the annotation target is not too thin.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/spring/physics_spring_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, scene variant, accent color, target answer, scale factor, and verifier payloads must be explicit in the instance trace.
- The task compares only the two visible extension markers, not the weight blocks or ruler tick labels.
