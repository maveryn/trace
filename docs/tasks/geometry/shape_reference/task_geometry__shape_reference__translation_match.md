# `task_geometry__shape_reference__translation_match`

## Contract
1. Domain: `geometry`
2. Scene id: `shape_reference`
5. Query id: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `point_set`

## Program Contract
- `label(select_option(candidate_shapes, transform_rule=translation_match)); scene=shape_reference; scope=translation_match`

## Reasoning Operations

Families: `transformation`, `matching`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `shape_reference`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is a `point_set` containing the winning polygon's visible vertices in pixel coordinates. The reference polygon, transformation cue, graph coordinates, labels, and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/shape_reference.yaml`
- Task module: `src/trace_tasks/tasks/geometry/shape_reference/translation_match.py`
