# `task_illustrations__isometric_quarry__terrain_elevation_extremum_label`

## Summary
- Domain: `illustrations`
- Scene id: `isometric_quarry`
- Implementation source scene: `isometric_quarry`
- Implementation source: `src/trace_tasks/tasks/illustrations/isometric_quarry/terrain_elevation_extremum_label.py`

## Task Contract
Selects the lettered terrain tile at the requested elevation extremum in an isometric quarry scene. Each scene samples an active elevation range from `0..1` or `0..2`.

## Program Contract

Program: `select(label, extremum(level(tile), mode=highest|lowest), tile in lettered_ground_tiles); scene=isometric_quarry; scope=terrain_elevation_extremum_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `terrain_elevation_extremum_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `extremum`, `level`, `tile`, `mode`, `highest`, `lowest`, `lettered_ground_tiles`, `isometric_quarry`, `terrain_elevation_extremum_label` plus the active `query_id` branch.
Operation: evaluate `select` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `highest_terrain_tile`, `lowest_terrain_tile`.

## Reasoning Operations

Families: `ranking`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `highest_terrain_tile` | `select(label, max(level(tile)), tile in lettered_ground_tiles); scene=isometric_quarry; scope=terrain_elevation_extremum_label` |
| `lowest_terrain_tile` | `select(label, min(level(tile)), tile in lettered_ground_tiles); scene=isometric_quarry; scope=terrain_elevation_extremum_label` |

## Program Metadata
- Program signatures: `select.spatial_elevation_extremum`
- Base program contract: `select(label, extremum(level(tile), mode=highest|lowest), tile in lettered_ground_tiles); scene=isometric_quarry; scope=terrain_elevation_extremum_label`
- Parameter axes: `canvas_profile`, `candidate_count=4`, `candidate_tile_ids`, `active_level_range`, `layout_family`, `terrain_level`, `extremum_mode`
- Arguments:
  - `tile`: visible lettered rock terrain tile; allowed generated candidate terrain tiles are plain quarry rock tiles not occupied by context objects, inside a same-level connected terrain region of at least six tiles, and supported by same-level neighbors on both grid axes; source `scene_ir.tiles`
  - `level`: integer terrain elevation; allowed active subset of `0|1|2`; source `scene_ir.tiles`
  - `mode`: extremum operator; allowed `highest|lowest`; source `query_id`
- Argument metadata status: `curated`
- Supported query ids: `highest_terrain_tile`, `lowest_terrain_tile`
- `scalar_annotation_checked`: true

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer is the letter on the unique candidate terrain tile whose ground level is highest or lowest as requested by the query branch.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation contains one bounding box around the selected terrain tile, not the letter badge and not any contextual quarry object.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/isometric_quarry/illustrations_isometric_quarry_v0.json`.
- Public prompts must refer to ground or terrain tiles so object height is not confused with terrain elevation.
- Render-only attributes such as palette, canvas profile, active max level, layout family, quarry cut placement, quarry equipment placement, tile geometry, and label font must not be query ids.
- Tile levels, active levels, layout family, quarry cuts, context quarry objects, candidate tile ids by label, candidate levels by label, selected label, selected tile id, selected tile bbox, projection metadata, and the scalar bbox annotation must be recorded in the trace.
