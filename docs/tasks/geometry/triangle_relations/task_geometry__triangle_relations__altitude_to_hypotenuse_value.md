# `task_geometry__triangle_relations__altitude_to_hypotenuse_value`

## Contract
1. Domain: `geometry`
2. Scene id: `triangle_relations`
3. Supported `query_id` values: `altitude_from_split_hypotenuse`, `missing_projection_from_altitude`
4. Answer schema: `integer`
5. Annotation schema: `point_map`
6. Scalar annotation checked: true

## Program Contract
- `solve_formula(right_triangle_altitude_to_hypotenuse, target=altitude_or_projection_length, formula_schema=altitude_geometric_mean); scene=triangle_relations; scope=altitude_to_hypotenuse_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Semantics
- `altitude_from_split_hypotenuse` asks for the altitude length from the two visible hypotenuse projections.
- `missing_projection_from_altitude` asks for one hypotenuse projection from the altitude and the other projection.

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `triangle_relations`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses keyed pixel points for the labeled construction points `A`, `B`, `C`, and `D`. Segment labels, right-angle markers, and target role remain visible diagram content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/triangle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/triangle_relations/altitude_to_hypotenuse_value.py`
