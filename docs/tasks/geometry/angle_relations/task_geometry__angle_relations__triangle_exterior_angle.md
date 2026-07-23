# `task_geometry__angle_relations__triangle_exterior_angle`

## Contract
1. Domain: `geometry`
2. Scene id: `angle_relations`
4. Query id: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `point_map`

## Program Contract
- `derive_geometry_metric(visible_angle_relations_measurements, derivation_rule=triangle_exterior_angle, output_role=angle_measure); scene=angle_relations; scope=triangle_exterior_angle`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `angle_relations`.
- Prompt schema: external prompt bundle
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. The annotation is a keyed point map over exactly `ABC`, `BAC`, and `BCD`.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/angle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/angle_relations/triangle_exterior_angle.py`
