# `task_illustrations__isometric_harbor__shoreline_nearest_boat_label`

## Summary
- Domain: `illustrations`
- Scene id: `isometric_harbor`
- Implementation source scene: `isometric_harbor`
- Implementation source: `src/trace_tasks/tasks/illustrations/isometric_harbor/shoreline_nearest_boat_label.py`

## Task Contract
Selects the lettered open-water boat whose bow is closest to the shoreline in an isometric harbor scene.

## Program Contract

Program: `argmin_label(boat, shoreline_distance(bow(boat), shoreline)); scene=isometric_harbor; scope=shoreline_nearest_boat_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `shoreline_nearest_boat_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `boat`, `shoreline_distance`, `bow`, `shoreline`, `isometric_harbor`, `shoreline_nearest_boat_label`.
Operation: evaluate `argmin_label` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `argmin_label(boat, shoreline_distance(bow(boat), shoreline)); scene=isometric_harbor; scope=shoreline_nearest_boat_label` |

## Program Metadata
- Program signatures: `selection.spatial_distance_extremum`
- Base program contract: `argmin_label(boat, shoreline_distance(bow(boat), shoreline)); scene=isometric_harbor; scope=shoreline_nearest_boat_label`
- Parameter axes: `canvas_profile`, `selected_label`, `dock_position`, `candidate_boat_tile_placement`, `boat_palette`, `label_font_family`, `shoreline_tiles`, `water_palette`
- Arguments:
  - `boat`: lettered open-water candidate entity with `object_type=boat`; source `scene_ir.entities[]`
  - `label`: visible option letter attached to the boat; source `scene_ir.entities[].metadata.shoreline_candidate_label`
  - `shoreline_distance`: renderer-assigned row distance from the shoreline water row; source `scene_ir.entities[].metadata.shoreline_distance_tiles`
  - `shoreline`: land-water boundary at the top of the harbor water field; source `scene_ir.relations`
- Argument metadata status: `curated`
- Supported query ids: `single`
- `scalar_annotation_checked`: true

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- Answer range: one of `A`, `B`, `C`, `D`, `E`, `F`

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation contains the pixel bounding box around the selected boat, not the option letter badge.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/isometric_harbor/illustrations_isometric_harbor_v1.json`.
- Prompts must ask for the lettered boat closest to the shoreline or shore, not closest to the dock.
- Render-only attributes such as boat color, exact open-water tile, dock jitter, canvas profile, label font, and water/land palette must not be query ids.
