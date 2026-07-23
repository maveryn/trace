# `task_illustrations__environment__lit_window_count`

## Summary
- Domain: `illustrations`
- Scene id: `environment`
- Implementation scene: `environment`
- Implementation source: `src/trace_tasks/tasks/illustrations/environment/lit_window_count.py`

## Task Contract
Counts lit windows in rendered environment buildings.

## Program Contract

Program: `count(filter(building_windows, is_lit(window))); scene=environment; scope=lit_window_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `lit_window_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `building_windows`, `is_lit`, `window`, `environment`, `lit_window_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; an integer in the default range `1..6`, derived from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(filter(building_windows, is_lit(window))); scene=environment; scope=lit_window_count` |

## Program Metadata
- Program signatures: `count.single_attribute_membership`
- Base program contract: `count(filter(building_windows, is_lit(window))); scene=environment; scope=lit_window_count`
- Parameter axes: `fixed_query`
- Arguments:
  - `building_windows`: semantic_role; allowed `visible_building_windows`; source `program_schema_concrete`
  - `window`: semantic_role; allowed `window_instance`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is an integer in the default range `1..6`, derived from the same execution trace as the annotation.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel boxes, one around each counted lit window. Do not include labels, numeric annotations, or context-only regions.
- Counted lit-window bboxes are generated with a minimum side of `24px`.
- Annotation and answer must be projected from the same generated scene trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the illustrations prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, and verifier payloads must be explicit in the instance trace.
- Distractor/context text may be rendered only when it is part of the scene grammar and must not be treated as annotation unless it is the queried visual witness.
