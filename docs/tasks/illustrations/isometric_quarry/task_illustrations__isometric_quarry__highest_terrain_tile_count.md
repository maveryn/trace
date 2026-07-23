# `task_illustrations__isometric_quarry__highest_terrain_tile_count`

## Summary
- Domain: `illustrations`
- Scene id: `isometric_quarry`
- Implementation source scene: `isometric_quarry`
- Implementation source: `src/trace_tasks/tasks/illustrations/isometric_quarry/highest_terrain_tile_count.py`

## Task Contract
Counts the visible top-surface terrain tiles that make up the highest elevation layer in an isometric quarry scene.

## Program Contract

Program: `count(tile where level(tile)=max(levels)); scene=isometric_quarry; scope=highest_terrain_tile_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `highest_terrain_tile_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `tile`, `where`, `level`, `max`, `levels`, `isometric_quarry`, `highest_terrain_tile_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(tile where level(tile)=max(levels)); scene=isometric_quarry; scope=highest_terrain_tile_count` |

## Program Metadata
- Program signatures: `count.spatial_elevation_tile_layer`
- Base program contract: `count(tile where level(tile)=max(levels)); scene=isometric_quarry; scope=highest_terrain_tile_count`
- Parameter axes: `canvas_profile`, `target_count=4..8`, `active_level_range`, `layout_family=count_plateau`, `highest_layer_shape`
- Arguments:
  - `tile`: visible top-surface terrain tile; source `scene_ir.tiles`
  - `level`: integer terrain elevation; source `scene_ir.tiles`
- Argument metadata status: `curated`
- Supported query ids: `single`
- `scalar_annotation_checked`: true

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- Answer range: `4..8`

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation contains one bounding box around the whole highest-elevation top layer.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/isometric_quarry/illustrations_isometric_quarry_v0.json`.
- The renderer must reserve highest-layer tiles from quarry patches and context objects for this task.
- Counted tile ids, counted tile bboxes, target level, answer count, and scalar bbox annotation must be recorded in trace metadata.
