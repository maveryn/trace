# `task_illustrations__rpg_tactical_map__terrain_type_tile_count`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_tactical_map`
- Implementation source scene: `rpg_tactical_map`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_tactical_map/terrain_type_tile_count.py`

## Task Contract
Counts all visible terrain tiles of a requested special terrain type on a top-down tactical RPG map.

## Program Contract

Program: `count(tile where terrain_type(tile) == target_terrain); scene=rpg_tactical_map; scope=terrain_type_tile_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `terrain_type_tile_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `tile`, `where`, `terrain_type`, `target_terrain`, `rpg_tactical_map`, `terrain_type_tile_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(tile where terrain_type(tile) == target_terrain); scene=rpg_tactical_map; scope=terrain_type_tile_count` |

## Program Metadata
- Program signatures: `count.tiles_by_terrain_type`
- Base program contract: `count(tile where terrain_type(tile) == target_terrain); scene=rpg_tactical_map; scope=terrain_type_tile_count`
- Parameter axes: `target_terrain`, `terrain_layout`, `water_feature_style`, `canvas_profile`, `answer_count_range`
- Arguments:
  - `tile`: terrain_tile; visible terrain grid tile; source `scene_ir.tiles`
  - `target_terrain`: one of `forest|mountain|water`; source `query_spec.params.target_terrain`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the number of visible tiles whose terrain type matches the requested target terrain.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains one pixel bounding box for each counted terrain tile.
- Annotation excludes the blue unit, non-target terrain tiles, bridge tiles, road tiles, and grass tiles.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_tactical_map/illustrations_rpg_tactical_map_v0.json`.
- Public prompts must name the target terrain type clearly.
- The task has no semantic query branch beyond `single`; sampled target terrain, map layout, water feature style, terrain colors, and canvas profile are trace metadata, not public query ids.
- The default generated answer range is `1..8` so count targets are present and large forest-count cases are avoided.
- Counted tile ids, counted tile bboxes, target terrain, integer answer, and bbox-set annotation must be recorded in the trace.
