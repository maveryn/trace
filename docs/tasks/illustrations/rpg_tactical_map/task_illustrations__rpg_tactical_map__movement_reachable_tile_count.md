# `task_illustrations__rpg_tactical_map__movement_reachable_tile_count`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_tactical_map`
- Implementation source scene: `rpg_tactical_map`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_tactical_map/movement_reachable_tile_count.py`

## Task Contract
Counts all terrain tiles, excluding the blue unit's starting tile, that the blue unit can reach within a visible movement-point budget on a top-down tactical RPG map.

## Program Contract

Program: `count(tile where reachable_by_movement_budget(unit, tile, budget, terrain_costs) and tile != start_tile); scene=rpg_tactical_map; scope=movement_reachable_tile_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `movement_reachable_tile_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `tile`, `where`, `reachable_by_movement_budget`, `unit`, `budget`, `terrain_costs`, `start_tile`, `rpg_tactical_map`, `movement_reachable_tile_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `topology`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(tile where reachable_by_movement_budget(unit, tile, budget, terrain_costs) and tile != start_tile); scene=rpg_tactical_map; scope=movement_reachable_tile_count` |

## Program Metadata
- Program signatures: `count.reachable_tiles_under_movement_budget`
- Base program contract: `count(tile where reachable_by_movement_budget(unit, tile, budget, terrain_costs) and tile != start_tile); scene=rpg_tactical_map; scope=movement_reachable_tile_count`
- Parameter axes: `movement_budget`, `terrain_layout`, `water_feature_style`, `canvas_profile`, `answer_count_range`
- Arguments:
  - `unit`: blue_unit; the single reference unit visible in the scene; source `scene_ir.units`
  - `tile`: terrain_tile; visible terrain grid tile; source `scene_ir.tiles`
  - `start_tile`: the tile occupied by the blue unit; source `scene_ir.relations.start_tile_id`
  - `budget`: integer; allowed `2|3`; source `query_spec.params.movement_budget`
  - `terrain_costs`: mapping; `grass=1`, `road=1`, `bridge=1`, `forest=2`, `mountain=3`, `water=blocked`; source `scene_ir.relations.terrain_movement_costs`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the number of non-starting terrain tiles reachable by the blue unit using at most the movement budget.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains one pixel bounding box for each counted reachable terrain tile.
- Annotation excludes the blue unit's starting tile, the blue unit itself, unreachable tiles, blocked water tiles, and path traces.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_tactical_map/illustrations_rpg_tactical_map_v0.json`.
- Public prompts must state the movement budget, that the starting tile is excluded, the orthogonal movement rule, and terrain costs, including mountain cost `3`.
- The task has no semantic query branch beyond `single`; sampled map layout, water feature style, budget, terrain colors, and canvas profile are trace metadata, not public query ids.
- The default generated answer range is `2..8` so counted movement areas remain visually manageable.
- Counted tile ids, counted tile bboxes, shortest movement costs, movement budget, starting tile id, integer answer, and bbox-set annotation must be recorded in the trace.
