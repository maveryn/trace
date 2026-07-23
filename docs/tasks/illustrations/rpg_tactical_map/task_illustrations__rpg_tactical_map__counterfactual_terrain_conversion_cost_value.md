# `task_illustrations__rpg_tactical_map__counterfactual_terrain_conversion_cost_value`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_tactical_map`
- Implementation source scene: `rpg_tactical_map`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_tactical_map/counterfactual_terrain_conversion_cost_value.py`

## Task Contract
Computes the fewest movement-point cost for the blue unit to reach one marked target tile after the best single conversion of one water, mountain, or forest tile into a road tile.

## Program Contract

Program: `value(min_shortest_movement_cost_after_unique_one_tile_to_road_conversion(unit, marked_tile, terrain_costs)); scene=rpg_tactical_map; scope=counterfactual_terrain_conversion_cost_value`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `counterfactual_terrain_conversion_cost_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `min_shortest_movement_cost_after_unique_one_tile_to_road_conversion`, `unit`, `marked_tile`, `terrain_costs`, `rpg_tactical_map`, `counterfactual_terrain_conversion_cost_value`.
Operation: evaluate `value` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `topology`, `state_update`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `value(min_shortest_movement_cost_after_unique_one_tile_to_road_conversion(unit, marked_tile, terrain_costs)); scene=rpg_tactical_map; scope=counterfactual_terrain_conversion_cost_value` |

## Program Metadata
- Program signatures: `value.counterfactual_shortest_movement_cost_one_tile_to_road`
- Base program contract: `value(min_shortest_movement_cost_after_unique_one_tile_to_road_conversion(unit, marked_tile, terrain_costs)); scene=rpg_tactical_map; scope=counterfactual_terrain_conversion_cost_value`
- Parameter axes: `terrain_layout`, `water_feature_style`, `target_tile`, `changed_tile`, `answer_value_range`, `canvas_profile`
- Arguments:
  - `unit`: blue_unit; the single reference unit visible in the scene; source `scene_ir.units`
  - `marked_tile`: terrain_tile; the tile with the visible target marker; source `render_map.target_tile_id`
  - `changed_tile`: terrain_tile; the unique tile converted to road in the optimal counterfactual; source `render_map.changed_tile_id`
  - `terrain_costs`: mapping; `grass=1`, `road=1`, `bridge=1`, `forest=2`, `mountain=3`, `water=blocked`; source `scene_ir.relations.terrain_movement_costs`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the minimum movement-point cost after exactly one water, mountain, or forest tile is changed into road before movement.
- Default generated answer range is `2..6`.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation contains role-keyed pixel bounding boxes for `player_cell`, `target_cell`, and `changed_cell`.
- `changed_cell` is the unique tile whose conversion to road gives the best counterfactual cost.
- Annotation excludes path tiles that are not one of those three roles.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_tactical_map/illustrations_rpg_tactical_map_v0.json`.
- Public prompts must state the terrain movement costs, that movement is orthogonal, that water cannot be entered before conversion, and that one water, mountain, or forest tile may be changed into road before moving.
- The task has no semantic query branch beyond `single`; sampled map layout, water feature style, target tile, changed tile, answer range, terrain colors, and canvas profile are trace metadata, not public query ids.
- Target tile id, changed tile id, changed tile original terrain, role-keyed annotation tile ids, original target cost, counterfactual shortest path ids, counterfactual path entry costs, counterfactual movement costs, terrain costs, start tile id, and answer value must be recorded in the trace.
