# `task_illustrations__park_playground__playground_equipment_count`

## Summary
- Domain: `illustrations`
- Scene id: `park_playground`
- Implementation source: `src/trace_tasks/tasks/illustrations/park_playground/playground_equipment_count.py`

## Task Contract
Count visible playground equipment items of one sampled type.

## Program Contract

Program: `count(filter(playground_equipment, equipment_type(equipment)=target_equipment_type)); scene=park_playground; scope=playground_equipment_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `playground_equipment_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `playground_equipment`, `equipment_type`, `equipment`, `target_equipment_type`, `park_playground`, `playground_equipment_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer is the count of equipment items whose rendered equipment type equals `target_equipment_type`.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel points, one near the center of each counted equipment item.
- Annotation and answer are projected from the same generated scene trace.

## Prompt And Trace Requirements
- Prompt text comes from `src/trace_tasks/resources/prompts/illustrations/park_playground/illustrations_park_playground_v0.json`.
- Render randomness, sampled style, `target_equipment_type`, and verifier payloads are recorded in trace metadata.
