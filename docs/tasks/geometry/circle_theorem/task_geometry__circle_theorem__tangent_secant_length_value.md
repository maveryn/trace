# `task_geometry__circle_theorem__tangent_secant_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_theorem`
5. Query id: `single`
6. Answer schema: `integer_value`
7. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_circle_theorem_measurements, unknown_role=length_measure, formula_schema=tangent_secant_length); scene=circle_theorem; scope=tangent_secant_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `circle_theorem`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_theorem.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_theorem/tangent_secant_length_value.py`
