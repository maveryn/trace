# `task_illustrations__environment__feature_relation_object_count`

## Summary
- Domain: `illustrations`
- Scene id: `environment`
- Implementation scene: `environment`
- Implementation source: `src/trace_tasks/tasks/illustrations/environment/feature_relation_object_count.py`

## Task Contract
Counts foreground objects in a queried relation to a road or river feature.

## Program Contract

Program: `count(filter(scene_objects, relation_to_feature(object, target_linear_feature)=target_relation)); scene=environment; scope=feature_relation_object_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `feature_relation_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `scene_objects`, `relation_to_feature`, `object`, `target_linear_feature`, `target_relation`, `environment`, `feature_relation_object_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; a non-negative integer derived from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `above_feature`, `below_feature`, `on_feature`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `above_feature` | `count(filter(scene_objects, relation_to_feature(object, target_linear_feature)=above)); scene=environment; scope=feature_relation_object_count` |
| `below_feature` | `count(filter(scene_objects, relation_to_feature(object, target_linear_feature)=below)); scene=environment; scope=feature_relation_object_count` |
| `on_feature` | `count(filter(scene_objects, relation_to_feature(object, target_linear_feature)=on)); scene=environment; scope=feature_relation_object_count` |

## Program Metadata
- Program signatures: `count.relation_attribute`
- Base program contract: `count(filter(scene_objects, relation_to_feature(object, target_linear_feature)=target_relation)); scene=environment; scope=feature_relation_object_count`
- Parameter axes: `target_linear_feature_type`, `target_relation`, `theme_id`
- Arguments:
  - `object`: semantic_role; allowed `scene_object`; source `program_schema_concrete`
  - `scene_objects`: semantic_role; allowed `visible_scene_objects`; source `program_schema_concrete`
  - `target_linear_feature`: semantic_role; allowed `visible_linear_feature`; source `program_schema_concrete`
  - `target_relation`: semantic_role; allowed `above`, `below`, `on`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `above_feature`, `below_feature`, `on_feature`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is a non-negative integer derived from the same execution trace as the annotation.
- `above_feature`, `below_feature`, and `on_feature` use default count support `1..6` so object placements remain visually countable while preserving varied answer values.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel boxes, one around each counted foreground object. Do not include the queried road/river feature, labels, numeric annotations, or context-only regions.
- Counted object bboxes are generated with a minimum side of `24px`.
- Annotation and answer must be projected from the same generated scene trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the illustrations prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, and verifier payloads must be explicit in the instance trace.
- The relation predicate is selected by `query_id`. For river `on`, prompt text uses `in or on the river`.
