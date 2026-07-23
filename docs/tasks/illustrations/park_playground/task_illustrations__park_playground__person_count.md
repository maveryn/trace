# `task_illustrations__park_playground__person_count`

## Summary
- Domain: `illustrations`
- Scene id: `park_playground`
- Implementation source: `src/trace_tasks/tasks/illustrations/park_playground/person_count.py`

## Task Contract
Count every visible person in one rendered park/playground illustration.

## Program Contract

Program: `count(visible_people); scene=park_playground; scope=person_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `person_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `visible_people`, `park_playground`, `person_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer is the number of rendered people in the scene.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel points, one near the center of each visible person.
- Annotation and answer are projected from the same generated scene trace.

## Prompt And Trace Requirements
- Prompt text comes from `src/trace_tasks/resources/prompts/illustrations/park_playground/illustrations_park_playground_v0.json`.
- The prompt scene layer must not disclose the sampled person count.
- Render randomness, sampled style, activity mix, person count, and verifier payloads are recorded in trace metadata.
