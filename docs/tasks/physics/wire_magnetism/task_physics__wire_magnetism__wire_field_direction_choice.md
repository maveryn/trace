# `task_physics__wire_magnetism__wire_field_direction_choice`

## Summary
- Domain: `physics`
- Scene id: `wire_magnetism`
- Implementation scene: `wire_magnetism`
- Implementation source: `src/trace_tasks/tasks/physics/wire_magnetism/wire_field_direction_choice.py`

## Program Contract

Program: `option_letter(direction(right_hand_rule_around_page_perpendicular_wire(current_page_direction, point_p_position))); scene=wire_magnetism; scope=wire_field_direction_choice`

Candidate set: the visible page-perpendicular current-carrying wire, dot/cross current cue, point marker, and magnetic-field arrow options inside the `wire_field_direction_choice` objective scope.
Operands: `current_page_direction` (semantic_role, allowed `visible_dot_or_cross_current_symbol`, source `program_schema_concrete`); `point_p_position` (semantic_role, allowed `visible_marked_point_position_around_wire`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation is keyed because witness roles are distinct; keys include `wire_current` and `point_p`. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Task Contract
Selects the in-plane magnetic-field arrow direction at a marked point near a wire carrying current into or out of the page.

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(direction(right_hand_rule_around_page_perpendicular_wire(current_page_direction, point_p_position))); scene=wire_magnetism; scope=wire_field_direction_choice` |

## Program Metadata
- Program signatures: `physics.wire_field_direction_choice`
- Base program contract: `option_letter(direction(right_hand_rule_around_page_perpendicular_wire(current_page_direction, point_p_position))); scene=wire_magnetism; scope=wire_field_direction_choice`
- Parameter axes: `fixed_query`
- Arguments:
  - `current_page_direction`: semantic_role; allowed `visible_dot_or_cross_current_symbol`; source `program_schema_concrete`
  - `point_p_position`: semantic_role; allowed `visible_marked_point_position_around_wire`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation is keyed because witness roles are distinct; keys include `wire_current` and `point_p`.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/wire_magnetism/physics_wire_magnetism_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
