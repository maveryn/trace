# `task_geometry__triangle_relations__right_triangle_missing_side_value`

## Contract
1. Domain: `geometry`
2. Scene id: `triangle_relations`
5. Query id: `single`
6. Answer schema: `number`
7. Annotation schema: `segment`
8. Scalar annotation checked: true

## Program Contract
- `solve_formula(visible_triangle_relations_measurements, unknown_role=side_length, formula_schema=right_triangle_trig_missing_side); scene=triangle_relations; scope=right_triangle_missing_side_value`
- The visible right-triangle construction shows one angle and one side measure sufficient to infer the requested target side. The target side role and trig formula family are internal trace metadata.

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `triangle_relations`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the requested visual segment as `[[x0,y0],[x1,y1]]`. Formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/triangle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/triangle_relations/right_triangle_missing_side_value.py`
