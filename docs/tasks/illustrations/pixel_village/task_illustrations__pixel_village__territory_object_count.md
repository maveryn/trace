# `task_illustrations__pixel_village__territory_object_count`

## Summary
- Domain: `illustrations`
- Scene id: `pixel_village`
- Implementation source scene: `pixel_village`
- Implementation source: `src/trace_tasks/tasks/illustrations/pixel_village/territory_object_count.py`

## Task Contract
Counts target entities inside one semantic pixel-village territory.

## Program Contract

Program: `count(filter(pixel_village_entities, territory_id(entity)=target_territory and public_name(entity)=target_public_name)); scene=pixel_village; scope=territory_object_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `territory_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `pixel_village_entities`, `territory_id`, `entity`, `target_territory`, `public_name`, `target_public_name`, `pixel_village`, `territory_object_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; a positive integer derived from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(filter(pixel_village_entities, territory_id(entity)=target_territory and public_name(entity)=target_public_name)); scene=pixel_village; scope=territory_object_count` |

## Program Metadata
- Program signatures: `count.scoped_attribute_membership`
- Base program contract: `count(filter(pixel_village_entities, territory_id(entity)=target_territory and public_name(entity)=target_public_name)); scene=pixel_village; scope=territory_object_count`
- Parameter axes: `territory_object`
- Supported operands: `cemetery_grave_marker`, `orchard_tree`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is a positive integer derived from the same execution trace as the annotation.
- Generated instances must keep the selected target count at or below the configured cap, currently `8`.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of final-image pixel boxes, one around each counted entity in the target territory.
- Annotation must not include whole territories, fences, paths, or context-only regions.

## Prompt And Trace Requirements
- Prompt text must come from the illustrations counting prompt bundle.
- Public prompts must name the territory and target public object type.
- The relevant territory is forced present for the sampled operand and the force constraint is recorded in trace metadata.
- Counted entity ids, territory id, target public name, renderer metadata, projected points, and diagnostic bboxes must be recorded in the trace.
