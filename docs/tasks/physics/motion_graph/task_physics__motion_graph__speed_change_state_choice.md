# `task_physics__motion_graph__speed_change_state_choice`

## Summary
- Domain: `physics`
- Scene id: `motion_graph`
- Implementation scene: `motion_graph`
- Implementation source: `src/trace_tasks/tasks/physics/motion_graph/speed_change_state_choice.py`

## Task Contract
Chooses the visible option letter describing the speed-change state represented by a marked interval on a velocity-time graph.

## Program Contract

Program: `option_letter(classify_speed_change(marked_velocity_time_graph_interval)); scene=motion_graph; scope=speed_change_state_choice`

Candidate set: the visible graph curve, marked interval, axis scales, labels, and option cells when present inside the `speed_change_state_choice` objective scope.
Operands: `marked_velocity_time_graph_interval` (semantic_role, allowed `highlighted_time_interval_with_local_velocity_curve_segment`, source `program_schema_concrete`); `motion_state` (query_operand, allowed `speeding_up|slowing_down|constant_speed`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the letter of the visible option describing whether the object is speeding up, slowing down, or moving at constant speed over the marked interval.
Annotation witnesses: `segment` witnesses from the finalized render. Annotation value: `[[x0, y0], [x1, y1]]` for the marked velocity-time graph segment, where each endpoint is a `[x, y]` pixel point. Annotation must mark the minimal visual segment witness for the marked interval. It must not mark axes, tick labels, option boxes, option letters, broad interval regions, or decorative graph chrome.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(classify_speed_change(marked_velocity_time_graph_interval)); scene=motion_graph; scope=speed_change_state_choice` |

## Program Metadata
- Program signatures: `physics.motion_graph_speed_change_state_choice`
- Base program contract: `option_letter(classify_speed_change(marked_velocity_time_graph_interval)); scene=motion_graph; scope=speed_change_state_choice`
- Parameter axes: `scene_variant`, `motion_state`, `correct_option_letter`
- Arguments:
  - `marked_velocity_time_graph_interval`: semantic_role; allowed `highlighted_time_interval_with_local_velocity_curve_segment`; source `program_schema_concrete`
  - `motion_state`: query_operand; allowed `speeding_up|slowing_down|constant_speed`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id`s: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the letter of the visible option describing whether the object is speeding up, slowing down, or moving at constant speed over the marked interval.

## Annotation Contract
- Annotation schema: `segment`
- Generator `annotation_gt.type`: `segment`
- Annotation value: `[[x0, y0], [x1, y1]]` for the marked velocity-time graph segment, where each endpoint is a `[x, y]` pixel point.
- Annotation must mark the minimal visual segment witness for the marked interval. It must not mark axes, tick labels, option boxes, option letters, broad interval regions, or decorative graph chrome.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics motion-graph v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- The public `query_id` is `single`; the trace records `motion_operation=speed_change_state_choice` as task-internal program metadata.
- Render randomness, sampled fonts/styles, marked interval, curve values, option mapping, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep axis labels, the marked interval, the local segment, and visual option boxes readable.
