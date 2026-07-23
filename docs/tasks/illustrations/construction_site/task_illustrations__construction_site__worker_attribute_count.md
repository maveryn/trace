# `task_illustrations__construction_site__worker_attribute_count`

## Summary
- Domain: `illustrations`
- Scene id: `construction_site`
- Implementation scene: `construction_site`
- Implementation source: `src/trace_tasks/tasks/illustrations/construction_site/worker_attribute_count.py`

## Contract
1. Domain: `illustrations`
2. Scene id: `construction_site`
3. Public task id: `task_illustrations__construction_site__worker_attribute_count`
4. Supported `query_id` values: `hard_hat_color_worker_count`, `vest_color_worker_count`
5. Query ids: `hard_hat_color_worker_count`, `vest_color_worker_count`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(filter(workers, worker_selector(worker, target_attribute, target_attribute_value))); scene=construction_site; scope=worker_attribute_count`

## Program Contract

Program: `count(filter(workers, worker_selector(worker, target_attribute, target_attribute_value))); scene=construction_site; scope=worker_attribute_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `worker_attribute_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `workers`, `worker_selector`, `worker`, `target_attribute`, `target_attribute_value`, `construction_site`, `worker_attribute_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; a non-negative integer derived from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `hard_hat_color_worker_count`, `vest_color_worker_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Task Contract
Counts visible workers matching one sampled safety-gear color attribute.

## Query Branches

| Query id | Program schema |
| --- | --- |
| `hard_hat_color_worker_count` | `count(filter(workers, worker_selector(worker, target_attribute, target_attribute_value))); scene=construction_site; scope=worker_attribute_count; query_branch=hard_hat_color_worker_count` |
| `vest_color_worker_count` | `count(filter(workers, worker_selector(worker, target_attribute, target_attribute_value))); scene=construction_site; scope=worker_attribute_count; query_branch=vest_color_worker_count` |

## Program Metadata
- Program signatures: `count.single_attribute_membership`
- Base program contract: `count(filter(workers, worker_selector(worker, target_attribute, target_attribute_value))); scene=construction_site; scope=worker_attribute_count`
- Parameter axes: `target_attribute`
- Arguments:
  - `target_attribute`: object_attribute; allowed `hard_hat_color`, `vest_color`; source `program_schema_concrete|query_id|parameter_axes`
  - `target_attribute_value`: object_attribute; allowed `sampled_color`; source `program_schema_concrete`
  - `worker`: semantic_role; allowed `worker_instance`; source `program_schema_concrete`
  - `workers`: semantic_role; allowed `visible_workers`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `hard_hat_color_worker_count`, `vest_color_worker_count`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- Configured answer support is `0..5`.
- The answer value is a non-negative integer derived from the same execution trace as the annotation.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel points, one near the center of each counted worker; use an empty set when the answer is `0`. Do not include labels, numeric annotations, or context-only regions.
- Annotation and answer must be projected from the same generated scene trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the `illustrations_construction_site_v1` scene prompt bundle, with scene/task/output layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, and verifier payloads must be explicit in the instance trace.
- Distractor/context text may be rendered only when it is part of the scene grammar and must not be treated as annotation unless it is the queried visual witness.
