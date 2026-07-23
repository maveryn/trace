# `task_illustrations__isometric_farmstead__highest_terrain_tile_count`

## Summary
- Domain: `illustrations`
- Scene id: `isometric_farmstead`
- Implementation source scene: `isometric_farmstead`
- Implementation source: `src/trace_tasks/tasks/illustrations/isometric_farmstead/highest_terrain_tile_count.py`

## Task Contract
Counts the visible top-surface terrain tiles that make up the highest elevation layer in an isometric farmstead.

## Program Contract

Program: `count(tile, level(tile)=max(level(terrain_tiles)) and tile in visible_top_surface_tiles); scene=isometric_farmstead; scope=highest_terrain_tile_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `highest_terrain_tile_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `tile`, `level`, `max`, `terrain_tiles`, `visible_top_surface_tiles`, `isometric_farmstead`, `highest_terrain_tile_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `logical_composition`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(tile, level(tile)=max(level(terrain_tiles)) and tile in visible_top_surface_tiles); scene=isometric_farmstead; scope=highest_terrain_tile_count` |

## Program Metadata
- Program signatures: `count.spatial_elevation_tiles`
- Base program contract: `count(tile, level(tile)=max(level(terrain_tiles)) and tile in visible_top_surface_tiles); scene=isometric_farmstead; scope=highest_terrain_tile_count`
- Parameter axes: `canvas_profile`, `target_count=4..8`, `highest_layer_shape`, `active_level_range`, `layout_family`
- Arguments:
  - `terrain_tiles`: all visible terrain tile top faces; source `scene_ir.tiles`
  - `level`: integer terrain elevation; source `scene_ir.tiles`
  - `highest_layer`: the connected set of visible terrain tiles at the maximum level; source `scene_ir.tiles`
- Argument metadata status: `curated`
- Supported query ids: `single`
- Internal prompt/query key: `highest_terrain_tile_count`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the number of visible top-surface terrain tiles on the highest elevation layer.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation contains one bounding box around the whole highest-elevation top layer. It is not a per-tile bbox set, and it must not include lower-level side faces as the intended witness.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/isometric_farmstead/illustrations_isometric_farmstead_v0.json`.
- Public prompts must ask for the count of visible tiles on the highest elevation layer.
- The renderer must keep the highest layer as one connected clean plateau and reserve highest-layer tiles from farm patches and context objects.
- Target level, counted tile ids, counted tile bboxes, highest-layer bbox, answer count, projection metadata, and the scalar bbox annotation must be recorded in the trace.
