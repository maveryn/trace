# `task_illustrations__rpg_dungeon__safe_reachable_chest_count`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_dungeon`
- Implementation source scene: `rpg_dungeon`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_dungeon/safe_reachable_chest_count.py`

## Task Contract
Counts treasure chests that are reachable from the player through unblocked dungeon paths and are not located in a chamber containing a monster.

## Program Contract

Program: `count(chest, reachable(chest_tile, player_tile, passable_tile=open_floor) and not exists(monster in chest.chamber)); scene=rpg_dungeon; scope=safe_reachable_chest_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `safe_reachable_chest_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `chest`, `reachable`, `chest_tile`, `player_tile`, `passable_tile`, `open_floor`, `exists`, `monster`, `chamber`, `rpg_dungeon`, `safe_reachable_chest_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `topology`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(chest, reachable(chest_tile, player_tile, passable_tile=open_floor) and not exists(monster in chest.chamber)); scene=rpg_dungeon; scope=safe_reachable_chest_count` |

## Program Metadata
- Program signatures: `count.filtered_reachable_graph_objects`
- Base program contract: `count(chest, reachable(chest_tile, player_tile, passable_tile=open_floor) and not exists(monster in chest.chamber)); scene=rpg_dungeon; scope=safe_reachable_chest_count`
- Parameter axes: `total_chest_count`, `safe_reachable_chest_count`, `reachable_chest_count`, `monster_chamber_count`, `reachable_monster_chamber_count`, `layout_orientation`, `side_counts`, `canvas_profile`
- Arguments:
  - `chest`: treasure_chest; allowed visible generated treasure chests; source `scene_ir.entities`
  - `player_tile`: tile coordinate containing the visible player; source `scene_ir.entities`
  - `passable_tile`: floor tile not occupied by a blocker; allowed open floor and corridors; source `program_schema_concrete`
  - `monster`: visible RPG dungeon monster; source `scene_ir.entities`
  - `safe_reachable_chest_count`: integer; allowed `0|1|2|3|4|5` and never greater than `total_chest_count - 1`; source `parameter_axes`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the number of reachable visible treasure chests whose chamber does not contain a monster.
- Each generated instance contains at least one boulder blocker and at least one monster, so the task requires both path reachability and chamber filtering.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains one bounding box around each counted reachable treasure chest and is empty when the answer is zero.
- Annotation excludes excluded reachable chests in monster chambers, unreachable chests, monsters, blockers, walls, floors, corridors, and background stone.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_dungeon/illustrations_rpg_dungeon_v0.json`.
- Public prompts must mention both unblocked/boulder-free reachability and excluding monster chambers.
- Public prompts use `chambers`, not `rooms`, for this scene.
- Render-only attributes such as palette, monster type, chamber positions, layout orientation, side counts, blocker type, and canvas profile must not be query ids.
- Floor tiles, graph edge ids, blocked edge ids, blocked tiles, layout orientation, side counts, total chest count, player entity, reachable chest ids, monster chamber ids, counted chest ids, projected bbox-set annotation, and diagnostic blocker/entity bboxes must be recorded in the trace.
