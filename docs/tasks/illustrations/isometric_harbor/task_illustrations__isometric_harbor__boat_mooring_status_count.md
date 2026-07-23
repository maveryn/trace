# `task_illustrations__isometric_harbor__boat_mooring_status_count`

## Summary
- Domain: `illustrations`
- Scene id: `isometric_harbor`
- Implementation source scene: `isometric_harbor`
- Implementation source: `src/trace_tasks/tasks/illustrations/isometric_harbor/boat_mooring_status_count.py`

## Task Contract
Counts boats by mooring status in an isometric harbor scene: tied along the main dock versus floating in open water away from the dock.

## Program Contract

Program: `count(boat where mooring_status(boat, main_dock)=status); scene=isometric_harbor; scope=boat_mooring_status_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `boat_mooring_status_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `boat`, `where`, `mooring_status`, `main_dock`, `status`, `isometric_harbor`, `boat_mooring_status_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `moored_boat_count`, `open_water_boat_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `moored_boat_count` | `count(boat where mooring_status(boat, main_dock)=moored); scene=isometric_harbor; scope=boat_mooring_status_count` |
| `open_water_boat_count` | `count(boat where mooring_status(boat, main_dock)=open_water); scene=isometric_harbor; scope=boat_mooring_status_count` |

## Program Metadata
- Program signatures: `count.spatial_relation_object_filter`
- Base program contract: `count(boat where mooring_status(boat, main_dock)=status); scene=isometric_harbor; scope=boat_mooring_status_count`
- Parameter axes: `canvas_profile`, `target_count`, `other_count`, `dock_position`, `boat_type`, `boat_palette`, `open_water_boat_orientation`, `dock_context_object_placement`, `shoreline_tiles`, `palette`
- Arguments:
  - `status`: mooring relation to the main dock; allowed `moored|open_water`; source `query_id`
  - `boat`: entity with `object_type=boat`; source `scene_ir.entities[]`
  - `mooring_status`: renderer-assigned metadata from the boat's placement; source `scene_ir.entities[].metadata.mooring_status`
- Argument metadata status: `curated`
- Supported query ids: `moored_boat_count`, `open_water_boat_count`
- `scalar_annotation_checked`: not applicable; annotation schema is `bbox_set`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- Answer range: `1..6`

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains one bounding box for each counted boat; use `[]` when the answer is `0`.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/isometric_harbor/illustrations_isometric_harbor_v1.json`.
- Prompts must distinguish boats tied along the main dock from boats floating in open water away from the dock.
- Render-only attributes such as boat style, boat orientation, dock jitter, canvas profile, shoreline placement, cargo placement, dock-post placement, and water/land palette must not be query ids.
