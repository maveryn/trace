# `task_illustrations__indoor_room__furniture_side_count`

## Summary
- Domain: `illustrations`
- Scene id: `indoor_room`
- Implementation source scene: `indoor_room`
- Implementation source: `src/trace_tasks/tasks/illustrations/indoor_room/furniture_side_count.py`

## Task Contract
Counts visible small indoor objects of a sampled object type that lie to the left or right of the table.

## Program Contract

Program: `count(filter(visible_room_objects, object_type(object)=target_object_type and side_relation(object, target_furniture)=query_side)); scene=indoor_room; scope=furniture_side_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `furniture_side_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `visible_room_objects`, `object_type`, `object`, `target_object_type`, `side_relation`, `target_furniture`, `query_side`, `indoor_room`, `furniture_side_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; a non-negative integer derived from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `left_side`, `right_side`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `left_side` | `count(filter(visible_room_objects, object_type(object)=target_object_type and left_of(object, target_furniture))); scene=indoor_room; scope=furniture_side_count` |
| `right_side` | `count(filter(visible_room_objects, object_type(object)=target_object_type and right_of(object, target_furniture))); scene=indoor_room; scope=furniture_side_count` |

## Program Metadata
- Program signatures: `count.spatial_relation_attribute`
- Base program contract: `count(filter(visible_room_objects, object_type(object)=target_object_type and side_relation(object, target_furniture)=query_side)); scene=indoor_room; scope=furniture_side_count`
- Parameter axes: `query_side`, `target_object_type`
- Arguments:
  - `visible_room_objects`: semantic_role; allowed `visible_room_objects`; source `program_schema_concrete`
  - `query_side`: relation; allowed `left_side|right_side`; source `query_id`
  - `target_object_type`: object_type; allowed `sampled_indoor_object_type`; source `trace_metadata`
  - `target_furniture`: furniture_type; allowed `table`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `left_side`, `right_side`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is a non-negative integer derived from the same execution trace as the annotation.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel boxes, one around each counted object satisfying the target type and side relation.
- Annotation excludes the reference furniture, labels, numeric annotations, and distractor/context objects.

## Prompt And Trace Requirements
- Prompt text must come from the indoor-room prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled styles, query side, target furniture, target object type, and verifier payloads must be explicit in the instance trace.
- Answer and annotation must be projected from the same generated scene trace, not inferred from pixels or prompt text.
