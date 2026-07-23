# `task_illustrations__rpg_dungeon__reachable_chest_count`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_dungeon`
- Implementation source scene: `rpg_dungeon`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_dungeon/reachable_chest_count.py`

## Task Contract
Counts the treasure chests reachable from the player by following only unblocked dungeon floor and corridor tiles.

## Program Contract

Program: `count(chest, reachable(chest_tile, player_tile, passable_tile=open_floor) and chest.object_type=treasure_chest); scene=rpg_dungeon; scope=reachable_chest_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `reachable_chest_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `chest`, `reachable`, `chest_tile`, `player_tile`, `passable_tile`, `open_floor`, `object_type`, `treasure_chest`, `rpg_dungeon`, `reachable_chest_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `topology`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(chest, reachable(chest_tile, player_tile, passable_tile=open_floor) and chest.object_type=treasure_chest); scene=rpg_dungeon; scope=reachable_chest_count` |

## Program Metadata
- Program signatures: `count.reachable_graph_objects`
- Base program contract: `count(chest, reachable(chest_tile, player_tile, passable_tile=open_floor) and chest.object_type=treasure_chest); scene=rpg_dungeon; scope=reachable_chest_count`
- Parameter axes: `player_tile`, `total_chest_count`, `reachable_chest_count`, `layout_orientation`, `side_counts`, `blocked_tiles`, `blocked_edge_ids`, `canvas_profile`
- Arguments:
  - `chest`: treasure_chest; allowed visible generated treasure chests; source `scene_ir.entities`
  - `player_tile`: tile coordinate containing the visible player; source `scene_ir.entities`
  - `passable_tile`: floor tile not occupied by a blocker; allowed open floor and corridors; source `program_schema_concrete`
  - `total_chest_count`: integer; allowed `4|5|6`; source `parameter_axes`
  - `reachable_chest_count`: integer; allowed `0|1|2|3|4|5|6` and never greater than `total_chest_count`; source `parameter_axes`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the number of visible treasure chests reachable from the player without crossing boulders or wall/background tiles. Each scene contains four to six visible treasure chests.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains one bounding box around each counted reachable treasure chest and is empty when the answer is zero.
- Annotation excludes unreachable chests, blockers, walls, and background stone.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_dungeon/illustrations_rpg_dungeon_v0.json`.
- Public prompts refer to the player and require following only unblocked/open floor paths.
- Render-only attributes such as palette, chamber positions, layout orientation, side counts, blocker type, and canvas profile must not be query ids.
- Floor tiles, graph edge ids, blocked edge ids, blocked tiles, layout orientation, side counts, total chest count, player entity, all chest entities, reachable chest ids, projected bbox-set annotation, and diagnostic blocker/entity bboxes must be recorded in the trace.
