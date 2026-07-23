# `task_illustrations__rpg_house__adjacent_room_count`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_house`
- Implementation source scene: `rpg_house`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_house/adjacent_room_count.py`

## Task Contract
Counts the rooms that directly share a doorway with the room containing the player in a top-down pixel RPG house layout.

## Program Contract

Program: `count(room, adjacent(room, player_room, relation=shares_doorway) and room != player_room); scene=rpg_house; scope=adjacent_room_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `adjacent_room_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `room`, `adjacent`, `player_room`, `shares_doorway`, `rpg_house`, `adjacent_room_count`.
Operation: evaluate `count` over rooms that directly share any doorway with the player's room; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `topology`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(room, adjacent(room, player_room, relation=shares_doorway) and room != player_room); scene=rpg_house; scope=adjacent_room_count` |

## Program Metadata
- Program signatures: `count.adjacent_rooms`
- Base program contract: `count(room, adjacent(room, player_room, relation=shares_doorway) and room != player_room); scene=rpg_house; scope=adjacent_room_count`
- Parameter axes: `player_room`, `adjacent_room_count`, `room_count`
- Arguments:
  - `room`: enclosed_room; allowed visible generated room regions; source `scene_ir.rooms`
  - `player_room`: room_id; allowed any generated room whose doorway degree matches the sampled answer; source `parameter_axes`
  - `shares_doorway`: room_relation; allowed direct doorway adjacency regardless of door open/closed state; source `program_schema_concrete`
  - `adjacent_room_count`: integer; allowed `1|2|3|4|5` when feasible for the generated room graph; source `parameter_axes`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the number of rooms other than the player's current room that directly share a doorway with it.

## Annotation Contract
- Annotation schema: `point_set_map`
- Generator `annotation_gt.type`: `point_set_map`
- Annotation key `player` contains one point on the visible player marker.
- Annotation key `adjacent_rooms` contains one center point for each counted directly adjacent room.
- Annotation excludes non-adjacent rooms, doors, furniture, walls, and decorative fixtures.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_house/illustrations_rpg_house_v0.json`.
- Public prompts refer to the room with the player and count only rooms that are one doorway away.
- Render-only attributes such as palette, floor colors, furniture layout, room shapes, door states, and canvas profile must not be query ids.
- Room graph, player room id, adjacent room ids, player point, adjacent room center points, generated room count, projected keyed point-set annotation, and diagnostic room/door bboxes must be recorded in the trace.
