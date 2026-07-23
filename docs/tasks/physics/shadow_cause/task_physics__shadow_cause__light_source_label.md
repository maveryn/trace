# `task_physics__shadow_cause__light_source_label`

## Summary
- Domain: `physics`
- Scene id: `shadow_cause`
- Implementation scene: `shadow_cause`
- Implementation source: `src/trace_tasks/tasks/physics/shadow_cause/light_source_label.py`

## Program Contract

Program: `option_letter(select(candidate_light_sources, direction_from_object_to_light_source=opposite(direction_from_object_to_cast_shadow))); scene=shadow_cause; scope=light_source_label`

Candidate set: the visible object, shadow geometry, light-source candidates, and option labels inside the `light_source_label` objective scope.
Operands: `candidate_light_sources` (visual_candidate_set, allowed `six_labeled_candidate_lamps`, source `program_schema_concrete`); `object` (semantic_role, allowed `visible_shadow_casting_object`, source `program_schema_concrete`); `cast_shadow` (semantic_role, allowed `visible_shadow_on_floor`, source `program_schema_concrete`); `shadow_direction` (query_operand, allowed `eight_compass_directions`, source `sampled_axis`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the label of the unique candidate light source opposite the shadow direction from the object.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation is one pixel box around the selected labeled light-source option. Annotation must mark the selected visible lamp option from the final rendered diagram. Object and shadow boxes remain trace metadata for derivation context.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(select(candidate_light_sources, direction_from_object_to_light_source=opposite(direction_from_object_to_cast_shadow))); scene=shadow_cause; scope=light_source_label` |

## Program Metadata
- Program signatures: `physics.shadow_cause_light_source_label`
- Base program contract: `option_letter(select(candidate_light_sources, direction_from_object_to_light_source=opposite(direction_from_object_to_cast_shadow))); scene=shadow_cause; scope=light_source_label`
- Parameter axes: `correct_option_letter`, `shadow_direction`, `object_shape`
- Arguments:
  - `candidate_light_sources`: visual_candidate_set; allowed `six_labeled_candidate_lamps`; source `program_schema_concrete`
  - `object`: semantic_role; allowed `visible_shadow_casting_object`; source `program_schema_concrete`
  - `cast_shadow`: semantic_role; allowed `visible_shadow_on_floor`; source `program_schema_concrete`
  - `shadow_direction`: query_operand; allowed `eight_compass_directions`; source `sampled_axis`
- Argument metadata status: `curated`
- Supported query ids: `single`
- Internal trace query branch: `source_from_shadow_label`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the label of the unique candidate light source opposite the shadow direction from the object.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation is one pixel box around the selected labeled light-source option.
- Annotation must mark the selected visible lamp option from the final rendered diagram. Object and shadow boxes remain trace metadata for derivation context.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/shadow_cause/physics_shadow_cause_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, shadow direction, object shape, option mapping, and verifier payloads must be explicit in the instance trace.
- Object color, object shape, and option letter must stay non-semantic; only the relative object-to-shadow direction determines the correct light source.
