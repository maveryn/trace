# `task_illustrations__rpg_house__reachable_room_count`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_house`
- Implementation source scene: `rpg_house`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_house/reachable_room_count.py`

## Task Contract
Counts the other rooms reachable from the player's room by following only open doors in a top-down pixel RPG house layout.

## Program Contract

Program: `count(room, reachable(room, player_room, passable_door_state=open) and room != player_room); scene=rpg_house; scope=reachable_room_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `reachable_room_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `room`, `reachable`, `player_room`, `passable_door_state`, `open`, `rpg_house`, `reachable_room_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `topology`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(room, reachable(room, player_room, passable_door_state=open) and room != player_room); scene=rpg_house; scope=reachable_room_count` |

## Program Metadata
- Program signatures: `count.reachable_component_rooms`
- Base program contract: `count(room, reachable(room, player_room, passable_door_state=open) and room != player_room); scene=rpg_house; scope=reachable_room_count`
- Parameter axes: `player_room`, `reachable_room_count`, `room_count`
- Arguments:
  - `room`: enclosed_room; allowed visible generated room regions; source `scene_ir.rooms`
  - `player_room`: room_id; allowed any generated room with enough graph capacity for the sampled answer; source `parameter_axes`
  - `passable_door_state`: door_state; allowed `open`; source `program_schema_concrete`
  - `reachable_room_count`: integer; allowed `0|1|2|3|4|5|6|7` when feasible for the generated room count; source `parameter_axes`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the number of rooms other than the player's current room that are reachable through open doorways.

## Annotation Contract
- Annotation schema: `point_set_map`
- Generator `annotation_gt.type`: `point_set_map`
- Annotation key `player` contains one point on the visible player marker.
- Annotation key `reachable_rooms` contains one center point for each counted reachable room and is empty when the answer is zero.
- Annotation excludes unreachable rooms, closed doors, furniture, walls, and decorative fixtures.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_house/illustrations_rpg_house_v0.json`.
- Public prompts refer to the room with the player and require following only open doors.
- Render-only attributes such as palette, floor colors, furniture layout, room shapes, and canvas profile must not be query ids.
- Room graph, door states, player room id, reachable room ids, player point, reachable room center points, generated room count, projected keyed point-set annotation, and diagnostic room/door bboxes must be recorded in the trace.
