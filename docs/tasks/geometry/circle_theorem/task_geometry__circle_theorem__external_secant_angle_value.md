# `task_geometry__circle_theorem__external_secant_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_theorem`
5. Query id: `single`
6. Answer schema: `integer_value`
7. Annotation schema: `point_map`

## Program Contract
- `derive_geometry_metric(visible_circle_theorem_measurements, derivation_rule=external_two_secants_angle_from_arcs, output_role=angle_measure); scene=circle_theorem; scope=external_secant_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `circle_theorem`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses keyed pixel points for the external point and the four circle intersection points. Arc labels and target angle marks are visible annotations plus verifier metadata, not separate annotation objects.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_theorem.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_theorem/external_secant_angle_value.py`
