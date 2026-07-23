# `task_illustrations__isometric_quarry__worker_same_level_tile_label`

## Summary
- Domain: `illustrations`
- Scene id: `isometric_quarry`
- Implementation source scene: `isometric_quarry`
- Implementation source: `src/trace_tasks/tasks/illustrations/isometric_quarry/worker_same_level_tile_label.py`

## Task Contract
Selects the lettered terrain tile at the same elevation level as the quarry worker.

## Program Contract

Program: `select(label, tile where level(tile)=level(reference_worker_tile), tile in lettered_ground_tiles); scene=isometric_quarry; scope=worker_same_level_tile_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `worker_same_level_tile_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `tile`, `where`, `level`, `reference_worker_tile`, `lettered_ground_tiles`, `isometric_quarry`, `worker_same_level_tile_label`.
Operation: evaluate `select` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select(label, tile where level(tile)=level(reference_worker_tile)); scene=isometric_quarry; scope=worker_same_level_tile_label` |

## Program Metadata
- Program signatures: `select.spatial_same_elevation_reference`
- Base program contract: `select(label, tile where level(tile)=level(reference_worker_tile), tile in lettered_ground_tiles); scene=isometric_quarry; scope=worker_same_level_tile_label`
- Parameter axes: `canvas_profile`, `candidate_count=4`, `candidate_tile_ids`, `reference_worker_tile_id`, `active_level_range`, `layout_family`
- Arguments:
  - `reference_worker_tile`: one worker on a safe rock tile; source `scene_ir.entities[role=reference]`
  - `tile`: visible lettered rock terrain tile; source `scene_ir.tiles`
  - `level`: integer terrain elevation; source `scene_ir.tiles`
- Argument metadata status: `curated`
- Supported query ids: `single`
- `scalar_annotation_checked`: true

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer is the unique candidate letter whose terrain level matches the worker's terrain level.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation contains one bounding box around the selected terrain tile, not the worker and not the letter badge.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/isometric_quarry/illustrations_isometric_quarry_v0.json`.
- Prompt/query metadata should use `worker_same_level_tile` as the semantic prompt key while public query id remains `single`.
- Candidate levels, worker tile id, worker bbox, selected tile id, selected label, and scalar bbox annotation must be recorded in trace metadata.
