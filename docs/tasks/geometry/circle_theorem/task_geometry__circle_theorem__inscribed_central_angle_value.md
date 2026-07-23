# `task_geometry__circle_theorem__inscribed_central_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_theorem`
5. Query id: `inscribed_angle_from_central`, `central_angle_from_inscribed`
6. Answer schema: `integer_value`
7. Annotation schema: `point_map`

## Program Contract
- `derive_geometry_metric(visible_circle_theorem_measurements, derivation_rule=central_inscribed_angle_relation, output_role=angle_measure); scene=circle_theorem; scope=inscribed_central_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches
- `inscribed_angle_from_central`: given the central angle over an arc, solve the inscribed angle over the same arc.
- `central_angle_from_inscribed`: given the inscribed angle over an arc, solve the central angle over the same arc.

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `circle_theorem`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_theorem.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_theorem/inscribed_central_angle_value.py`
