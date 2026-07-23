# `task_illustrations__pixel_village__river_side_object_count`

## Summary
- Domain: `illustrations`
- Scene id: `pixel_village`
- Implementation source scene: `pixel_village`
- Implementation source: `src/trace_tasks/tasks/illustrations/pixel_village/river_side_object_count.py`

## Task Contract
Counts visible target entities that lie strictly on one named side of the river.

## Program Contract

Program: `count(filter(pixel_village_entities, target_entity_type(entity)=target_object and strict_side_of_river(entity_tile_footprint, river_side))); scene=pixel_village; scope=river_side_object_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `river_side_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `pixel_village_entities`, `target_entity_type`, `entity`, `target_object`, `strict_side_of_river`, `entity_tile_footprint`, `river_side`, `pixel_village`, `river_side_object_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; a positive integer derived from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(filter(pixel_village_entities, target_entity_type(entity)=target_object and strict_side_of_river(entity_tile_footprint, river_side))); scene=pixel_village; scope=river_side_object_count` |

## Program Metadata
- Program signatures: `count.spatial_relation_membership`
- Base program contract: `count(filter(pixel_village_entities, target_entity_type(entity)=target_object and strict_side_of_river(entity_tile_footprint, river_side))); scene=pixel_village; scope=river_side_object_count`
- Parameter axes: `target_object`, `river_side`
- Arguments:
  - `entity`: semantic_role; allowed `pixel_village_entity`; source `program_schema_concrete`
  - `pixel_village_entities`: semantic_role; allowed `visible_pixel_village_entities`; source `program_schema_concrete`
  - `target_object`: semantic_role; allowed `building`, `person`, `tree`; source `parameter_axes`
  - `river_side`: spatial_relation; allowed `left`, `right`, `above`, `below`; source `parameter_axes`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is a positive integer derived from the same execution trace as the annotation.
- Generated instances force a visible balanced river and keep the selected target count at or below the configured cap, currently `8`.
- Tree-count instances suppress cemetery territory so cemetery dead-tree decor is not an ambiguous non-counted witness.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel points, one near the center of each counted entity.
- Annotation must not include the river, bridge, paths, whole territories, or context-only regions.

## Prompt And Trace Requirements
- Prompt text must come from the illustrations counting prompt bundle.
- Public prompts use ordinary spatial language such as `left of the river`, not tile or water-bound terminology.
- For `left`/`right`, the renderer forces a vertical river; for `above`/`below`, it forces a horizontal river.
- An entity counts only when its full tile footprint lies strictly on the requested side of the river bounds.
- Render-only attributes such as gender, facing, tree subtype, door state, roof style, season, and territory styling must not be queried.
- Counted entity ids, target object, target public name, river side, river orientation, river bounds, renderer metadata, projected points, and diagnostic bboxes must be recorded in the trace.
