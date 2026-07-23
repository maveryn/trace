# `task_geometry__triangle_relations__similar_triangles_side_length`

## Contract
1. Domain: `geometry`
2. Scene id: `triangle_relations`
5. Query id: `single`
6. Answer schema: `integer_value`
7. Annotation schema: `segment`
8. Scalar annotation checked: true

## Program Contract
- `solve_formula(visible_triangle_relations_measurements, unknown_role=length_measure, formula_schema=similar_triangles_side_length); scene=triangle_relations; scope=similar_triangles_side_length`
- The visible construction marks and prompt state that `DE` is parallel to `BC`.

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `triangle_relations`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the requested visual segment as `[[x0,y0],[x1,y1]]`. Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/triangle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/triangle_relations/similar_triangles_side_length.py`
