# `task_illustrations__environment__crossing_feature_count`

## Summary
- Domain: `illustrations`
- Scene id: `environment`
- Implementation scene: `environment`
- Implementation source: `src/trace_tasks/tasks/illustrations/environment/crossing_feature_count.py`

## Task Contract
Counts visible features that cross a queried road or river feature.

## Program Contract

Program: `count(filter(environment_features, crosses(feature, target_linear_feature))); scene=environment; scope=crossing_feature_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `crossing_feature_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `environment_features`, `crosses`, `feature`, `target_linear_feature`, `environment`, `crossing_feature_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; a non-negative integer derived from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(filter(environment_features, crosses(feature, target_linear_feature))); scene=environment; scope=crossing_feature_count` |

## Program Metadata
- Program signatures: `count.relation_attribute`
- Base program contract: `count(filter(environment_features, crosses(feature, target_linear_feature))); scene=environment; scope=crossing_feature_count`
- Parameter axes: `crossing_type`, `theme_id`
- Arguments:
  - `environment_features`: semantic_role; allowed `visible_environment_features`; source `program_schema_concrete`
  - `feature`: semantic_role; allowed `environment_feature_instance`; source `program_schema_concrete`
  - `target_linear_feature`: semantic_role; allowed `visible_linear_feature`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is a non-negative integer derived from the same execution trace as the annotation.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel points, one near the center of each counted/selected visual witness. Do not include labels, numeric annotations, or context-only regions.
- Annotation and answer must be projected from the same generated scene trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the illustrations prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, and verifier payloads must be explicit in the instance trace.
- Distractor/context text may be rendered only when it is part of the scene grammar and must not be treated as annotation unless it is the queried visual witness.
