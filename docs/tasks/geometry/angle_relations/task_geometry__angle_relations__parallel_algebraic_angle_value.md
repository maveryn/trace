# `task_geometry__angle_relations__parallel_algebraic_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `angle_relations`
4. Query ids: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `point_map`

## Program Contract
- `derive_geometry_metric(visible_parallel_line_angle_expressions, derivation_rule=same_side_supplementary_parallel_transversal_algebra, output_role=target_angle_measure); scene=angle_relations; scope=parallel_algebraic_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `angle_relations`.
- Prompt schema: external prompt bundle
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. The annotation is a keyed point map over exactly the visible angle names `BPQ`, `DQP`, and `FRQ`; each value is the angle vertex point. The displayed expressions on `BPQ` and `DQP` determine `x` by the same-side supplementary relation, and `FRQ` is the target angle marked with `?`.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/angle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/angle_relations/parallel_algebraic_angle_value.py`
