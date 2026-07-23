# `task_geometry__triangle_relations__angle_bisector_variable_value`

## Contract
1. Domain: `geometry`
2. Scene id: `triangle_relations`
5. Query id: `single`
6. Answer schema: `integer`
7. Annotation schema: `point_map`
8. Scalar annotation checked: true

## Program Contract
- `solve_formula(angle_bisector_theorem_variable, unknown_role=variable_value, formula_schema=angle_bisector_side_split_ratio); scene=triangle_relations; scope=angle_bisector_variable_value`
- The visible construction marks and prompt state that `AD` bisects angle `BAC`.

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `triangle_relations`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses keyed pixel points for the labeled construction points `A`, `B`, `C`, and `D`. Expression labels, tick marks, and solved variable values remain visible diagram content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/triangle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/triangle_relations/angle_bisector_variable_value.py`
