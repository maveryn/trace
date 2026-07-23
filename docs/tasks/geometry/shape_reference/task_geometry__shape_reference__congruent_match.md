# `task_geometry__shape_reference__congruent_match`

## Contract
1. Domain: `geometry`
2. Scene id: `shape_reference`
5. Query id: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `point_set`

## Program Contract
- `label(select_option(candidate_shapes, shape_relation(shape, reference_shape)=congruent_to_reference)); scene=shape_reference; scope=congruent_match`

## Reasoning Operations

Families: `matching`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `shape_reference`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is a `point_set` containing the selected polygon's visible vertices in pixel coordinates. The reference shape, labels, relation rule, graph coordinates, and symbolic shape metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/shape_reference.yaml`
- Task module: `src/trace_tasks/tasks/geometry/shape_reference/congruent_match.py`
