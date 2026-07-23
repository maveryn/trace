# `task_physics__switch_circuit__lit_bulb_count`

## Summary
- Domain: `physics`
- Scene id: `switch_circuit`
- Implementation source: `src/trace_tasks/tasks/physics/switch_circuit/lit_bulb_count.py`

## Task Contract
Counts how many labeled bulbs will be on in a mixed branch battery circuit with visibly open and closed switches.

## Program Contract

Program: `count(filter(bulbs, lies_on_complete_path(bulb_edge, battery_positive, battery_negative, closed_switches))); scene=switch_circuit; scope=lit_bulb_count`

Candidate set: the visible switch states, bulb symbols, wire topology, and bulb labels inside the `lit_bulb_count` objective scope.
Operands: `bulbs` (visual_candidate_set, allowed `five_labeled_bulbs_b1_through_b5`, source `program_schema_concrete`); `closed_switches` (semantic_role, allowed `visible_open_closed_switches_s1_through_s5`, source `program_schema_concrete`); `mixed_branch_topology` (semantic_role, allowed `parallel_branches_with_local_sub_branch`, source `program_schema_concrete`); `battery_positive_negative` (semantic_role, allowed `visible_ideal_battery_terminals`, source `program_schema_concrete`).
Operation: evaluate `count` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the count of bulbs whose bulb edge lies on at least one complete path from battery positive to battery negative through closed switches. Supported answers are `0..5`.
Annotation witnesses: `bbox_set` witnesses from the finalized render. Annotation is the unordered set of bboxes around bulb symbols that will be on. If the answer is `0`, annotation is an empty array.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(filter(bulbs, lies_on_complete_path(bulb_edge, battery_positive, battery_negative, closed_switches))); scene=switch_circuit; scope=lit_bulb_count; query_branch=lit_bulb_count` |

## Program Metadata
- Program signatures: `physics.switch_circuit_lit_bulb_count`
- Base program contract: `count(filter(bulbs, lies_on_complete_path(bulb_edge, battery_positive, battery_negative, closed_switches))); scene=switch_circuit; scope=lit_bulb_count`
- Parameter axes: `target_answer`, `switch_states`, `accent_color_name`
- Arguments:
  - `bulbs`: visual_candidate_set; allowed `five_labeled_bulbs_b1_through_b5`; source `program_schema_concrete`
  - `closed_switches`: semantic_role; allowed `visible_open_closed_switches_s1_through_s5`; source `program_schema_concrete`
  - `mixed_branch_topology`: semantic_role; allowed `parallel_branches_with_local_sub_branch`; source `program_schema_concrete`
  - `battery_positive_negative`: semantic_role; allowed `visible_ideal_battery_terminals`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the count of bulbs whose bulb edge lies on at least one complete path from battery positive to battery negative through closed switches. Supported answers are `0..5`.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is the unordered set of bboxes around bulb symbols that will be on.
- If the answer is `0`, annotation is an empty array.
- Annotation must not mark switch bboxes, wires, battery chrome, labels alone, decorative grid lines, or derived current paths.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the `physics_switch_circuit_v1` prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, switch states, graph edges, lit bulb ids, and verifier payloads must be explicit in the instance trace.
- Bulbs must not visually glow or otherwise encode the answer; the count comes from switch connectivity.
