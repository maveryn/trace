# `task_geometry__triangle_relations__split_triangle_trig_side_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `triangle_relations`
3. Query id: `single`
4. Answer schema: `number`
5. Answer precision: `one_decimal`
6. Annotation schema: `point_map`
7. Scalar annotation checked: true

## Program Contract
- `solve_formula(shared_altitude_right_triangle_trig, unknown_role=target_side_length, formula_schema=shared_altitude_right_triangle_trig); scene=triangle_relations; scope=split_triangle_trig_side_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `triangle_relations`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses keyed pixel points for the labeled construction points `A`, `B`, `C`, and `D`. Side labels, angle labels, right-angle marks, equality ticks, and internal construction family remain visible diagram content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/triangle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/triangle_relations/split_triangle_trig_side_length_value.py`
