# `task_illustrations__indoor_room__surface_object_count`

## Summary
- Domain: `illustrations`
- Scene id: `indoor_room`
- Implementation source scene: `indoor_room`
- Implementation source: `src/trace_tasks/tasks/illustrations/indoor_room/surface_object_count.py`

## Task Contract
Counts visible small indoor objects of a sampled object type that are placed on one sampled room surface.

## Program Contract

Program: `count(filter(visible_room_objects, object_type(object)=target_object_type and on_surface(object, target_surface))); scene=indoor_room; scope=surface_object_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `surface_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `visible_room_objects`, `object_type`, `object`, `target_object_type`, `on_surface`, `target_surface`, `indoor_room`, `surface_object_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; an integer in `1..5`, derived from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(filter(visible_room_objects, object_type(object)=target_object_type and on_surface(object, target_surface))); scene=indoor_room; scope=surface_object_count` |

## Program Metadata
- Program signatures: `count.scoped_attribute`
- Base program contract: `count(filter(visible_room_objects, object_type(object)=target_object_type and on_surface(object, target_surface))); scene=indoor_room; scope=surface_object_count`
- Parameter axes: `target_object_type`, `target_surface`
- Arguments:
  - `visible_room_objects`: semantic_role; allowed `visible_room_objects`; source `program_schema_concrete`
  - `target_object_type`: object_type; allowed `sampled_indoor_object_type`; source `trace_metadata`
  - `target_surface`: surface_type; allowed `table|shelf|counter`; source `trace_metadata`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is an integer in `1..5`, derived from the same execution trace as the annotation.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel boxes, one around each counted object on the target surface.
- Annotation excludes the surface, labels, numeric annotations, and distractor/context objects.

## Prompt And Trace Requirements
- Prompt text must come from the indoor-room prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled styles, target object type, target surface, and verifier payloads must be explicit in the instance trace.
- Answer and annotation must be projected from the same generated scene trace, not inferred from pixels or prompt text.
