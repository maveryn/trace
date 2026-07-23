# `task_illustrations__rpg_house__room_count`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_house`
- Implementation source scene: `rpg_house`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_house/room_count.py`

## Task Contract
Counts all distinct enclosed rooms in a top-down pixel RPG house layout.

## Program Contract

Program: `count(room, enclosed(room)); scene=rpg_house; scope=room_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `room_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `room`, `enclosed`, `rpg_house`, `room_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(room, enclosed(room)); scene=rpg_house; scope=room_count` |

## Program Metadata
- Program signatures: `count.enclosed_rooms`
- Base program contract: `count(room, enclosed(room)); scene=rpg_house; scope=room_count`
- Parameter axes: `room_count`
- Arguments:
  - `room`: enclosed_room; allowed visible enclosed room regions; source `scene_ir.rooms`
  - `room_count`: integer; allowed `4|5|6|7|8`; source `parameter_axes`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the total number of distinct enclosed rooms visible in the generated house.

## Annotation Contract
- Annotation schema: `point_set`
- Generator `annotation_gt.type`: `point_set`
- Annotation contains one center point for each counted room.
- Annotation excludes doors, furniture, walls, and decorative fixtures.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_house/illustrations_rpg_house_v0.json`.
- Public prompts ask for the total number of enclosed rooms in the house layout.
- Render-only attributes such as palette, floor colors, door states, furniture layout, room names, and room shapes must not be query ids.
- Room ids, room bboxes, room center points, room count, room-count support, and projected point-set annotation must be recorded in the trace.
