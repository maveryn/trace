# `task_illustrations__rpg_tactical_map__movement_cost_value`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_tactical_map`
- Implementation source scene: `rpg_tactical_map`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_tactical_map/movement_cost_value.py`

## Task Contract
Computes the fewest movement-point cost for the blue unit to reach one visibly marked destination tile on a top-down tactical RPG map.

## Program Contract

Program: `value(shortest_movement_cost(unit, marked_tile, terrain_costs)); scene=rpg_tactical_map; scope=movement_cost_value`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `movement_cost_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `shortest_movement_cost`, `unit`, `marked_tile`, `terrain_costs`, `rpg_tactical_map`, `movement_cost_value`.
Operation: evaluate `value` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `topology`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `value(shortest_movement_cost(unit, marked_tile, terrain_costs)); scene=rpg_tactical_map; scope=movement_cost_value` |

## Program Metadata
- Program signatures: `value.shortest_movement_cost_to_marked_tile`
- Base program contract: `value(shortest_movement_cost(unit, marked_tile, terrain_costs)); scene=rpg_tactical_map; scope=movement_cost_value`
- Parameter axes: `terrain_layout`, `water_feature_style`, `target_tile`, `movement_cost_range`, `canvas_profile`
- Arguments:
  - `unit`: blue_unit; the single reference unit visible in the scene; source `scene_ir.units`
  - `marked_tile`: terrain_tile; the tile with the visible target marker; source `render_map.target_tile_id`
  - `terrain_costs`: mapping; `grass=1`, `road=1`, `bridge=1`, `forest=2`, `mountain=3`, `water=blocked`; source `scene_ir.relations.terrain_movement_costs`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the shortest total movement-point cost from the blue unit's starting tile to the marked target tile.
- Default answer range is `3..6`.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation contains exactly two role-keyed pixel bounding boxes:
  - `player_cell`: the blue unit's starting tile.
  - `target_cell`: the marked target tile.
- Annotation excludes the shortest path because multiple shortest paths can be valid for the same movement cost.
- Annotation excludes the target marker artwork outside tile boundaries.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_tactical_map/illustrations_rpg_tactical_map_v0.json`.
- Public prompts must state the terrain movement costs, that movement is orthogonal, and that water cannot be entered.
- The task has no semantic query branch beyond `single`; sampled map layout, water feature style, target tile, target movement cost, terrain colors, and canvas profile are trace metadata, not public query ids.
- Target tile id, target tile bbox, role-keyed annotation tile ids, shortest path tile ids for diagnostics, shortest path terrain labels, shortest path entry costs, shortest movement costs, terrain costs, start tile id, target Manhattan distance, and answer value must be recorded in the trace.
