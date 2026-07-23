# `task_illustrations__isometric_harbor__boat_side_count`

## Summary
- Domain: `illustrations`
- Scene id: `isometric_harbor`
- Implementation source scene: `isometric_harbor`
- Implementation source: `src/trace_tasks/tasks/illustrations/isometric_harbor/boat_side_count.py`

## Task Contract
Counts boats docked on the image-left or image-right side of the main dock in an isometric harbor scene.

## Program Contract

Program: `count(boat where dock_side(boat, main_dock)=side); scene=isometric_harbor; scope=boat_side_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `boat_side_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `boat`, `where`, `dock_side`, `main_dock`, `side`, `isometric_harbor`, `boat_side_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `left_side_boat_count`, `right_side_boat_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `left_side_boat_count` | `count(boat where dock_side(boat, main_dock)=image_left); scene=isometric_harbor; scope=boat_side_count` |
| `right_side_boat_count` | `count(boat where dock_side(boat, main_dock)=image_right); scene=isometric_harbor; scope=boat_side_count` |

## Program Metadata
- Program signatures: `count.spatial_side_object_filter`
- Base program contract: `count(boat where dock_side(boat, main_dock)=side); scene=isometric_harbor; scope=boat_side_count`
- Parameter axes: `canvas_profile`, `target_count`, `dock_position`, `boat_type`, `boat_palette`, `dock_context_object_placement`, `shoreline_tiles`, `palette`
- Arguments:
  - `side`: image-relative dock side; allowed `image_left|image_right`; source `query_id`
  - `boat`: entity with `object_type=boat`; source `scene_ir.entities[]`
  - `dock_side`: side metadata assigned by the renderer from each boat's mooring tile; source `scene_ir.entities[].metadata.dock_side`
- Argument metadata status: `curated`
- Supported query ids: `left_side_boat_count`, `right_side_boat_count`
- `scalar_annotation_checked`: not applicable; annotation schema is `bbox_set`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- Answer range: `0..5`

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains one bounding box for each counted boat; use `[]` when the answer is `0`.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/isometric_harbor/illustrations_isometric_harbor_v1.json`.
- Prompts must say `image-left side` or `image-right side` of the main dock to avoid world-frame ambiguity.
- Render-only attributes such as boat style, dock jitter, canvas profile, shoreline placement, cargo placement, dock-post placement, and water/land palette must not be query ids.
