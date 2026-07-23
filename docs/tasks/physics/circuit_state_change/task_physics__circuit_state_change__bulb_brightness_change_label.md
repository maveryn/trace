# `task_physics__circuit_state_change__bulb_brightness_change_label`

## Summary
- Domain: `physics`
- Scene id: `circuit_state_change`
- Implementation scene: `circuit_state_change`
- Implementation source: `src/trace_tasks/tasks/physics/circuit_state_change/bulb_brightness_change_label.py`

## Task Contract
Selects the labeled bulb whose brightness changes in the requested way after a visible switch action.

## Program Contract

Program: `label(select(bulbs, brightness_change_after_switch_action(bulb)=target_change_class)); scene=circuit_state_change; scope=bulb_brightness_change_label`

Candidate set: the visible switch action cue, bulb symbols, branch topology, and component labels inside the `bulb_brightness_change_label` objective scope.
Operands: `bulbs` (semantic_role, allowed `visible_labeled_bulbs_b1_through_b5_with_resistance_labels`, source `program_schema_concrete`); `switch_action` (semantic_role, allowed `opens|closes`, source `program_schema_concrete`); `target_change_class` (query_operand, allowed `brightens|dims|turns_on|turns_off`, source `query_id`); `circuit_topology` (semantic_role, allowed `series_bulb_plus_switch_controlled_parallel_branch_with_unchanged_reference_branch`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `label` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; The answer value is the selected visible bulb label, for example `B2`.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation is keyed by `changed_switch` and visible bulb labels `B1` through `B5`. Annotation marks the switch-action cue and bulb symbols with their resistance labels. It does not mark wires, battery terminals, decorative parts, inferred current paths, or derived brightness values.
Query ids: `brightens_after_switch_change`, `dims_after_switch_change`, `turns_on_after_switch_change`, `turns_off_after_switch_change`.

## Reasoning Operations

Families: `topology`, `state_update`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `brightens_after_switch_change` | `label(select(bulbs, brightness_change_after_switch_action(bulb)=brightens)); scene=circuit_state_change; scope=bulb_brightness_change_label; query_branch=brightens_after_switch_change` |
| `dims_after_switch_change` | `label(select(bulbs, brightness_change_after_switch_action(bulb)=dims)); scene=circuit_state_change; scope=bulb_brightness_change_label; query_branch=dims_after_switch_change` |
| `turns_on_after_switch_change` | `label(select(bulbs, brightness_change_after_switch_action(bulb)=turns_on)); scene=circuit_state_change; scope=bulb_brightness_change_label; query_branch=turns_on_after_switch_change` |
| `turns_off_after_switch_change` | `label(select(bulbs, brightness_change_after_switch_action(bulb)=turns_off)); scene=circuit_state_change; scope=bulb_brightness_change_label; query_branch=turns_off_after_switch_change` |

## Program Metadata
- Program signatures: `physics.circuit_state_change_bulb_brightness_change_label`
- Base program contract: `label(select(bulbs, brightness_change_after_switch_action(bulb)=target_change_class)); scene=circuit_state_change; scope=bulb_brightness_change_label`
- Parameter axes: `query_id`, `switch_action`, `resistance_values`, `target_label`, `accent_color_name`
- Arguments:
  - `bulbs`: semantic_role; allowed `visible_labeled_bulbs_b1_through_b5_with_resistance_labels`; source `program_schema_concrete`
  - `switch_action`: semantic_role; allowed `opens|closes`; source `program_schema_concrete`
  - `target_change_class`: query_operand; allowed `brightens|dims|turns_on|turns_off`; source `query_id`
  - `circuit_topology`: semantic_role; allowed `series_bulb_plus_switch_controlled_parallel_branch_with_unchanged_reference_branch`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `brightens_after_switch_change`, `dims_after_switch_change`, `turns_on_after_switch_change`, `turns_off_after_switch_change`

## Answer Contract
- Answer schema: `string`
- Generator `answer_gt.type`: `string`
- The answer value is the selected visible bulb label, for example `B2`.
- Each generated instance must have exactly one bulb matching the queried change class.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation is keyed by `changed_switch` and visible bulb labels `B1` through `B5`.
- Annotation marks the switch-action cue and bulb symbols with their resistance labels. It does not mark wires, battery terminals, decorative parts, inferred current paths, or derived brightness values.
- Annotation and answer are projected from the same generated execution trace.

## Prompt And Trace Requirements
- Prompt text comes from `src/trace_tasks/resources/prompts/physics/circuit_state_change/physics_circuit_state_change_v1.json`.
- Render randomness, sampled fonts/styles, switch action, visible resistance values, before/after power values, change classes, and verifier payloads are explicit in the instance trace.
- Bulbs must not visually glow or otherwise encode the answer; the selection comes from comparing the circuit before and after the switch action.
- This task remains separate from `task_physics__bulb_circuit__brightness_extremum_label`, which asks a static brightest/dimmest ranking.
