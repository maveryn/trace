# `task_geometry__triangle_relations__leg_projection_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `triangle_relations`
3. Supported `query_id` values: `leg_from_hypotenuse_projection`, `projection_from_leg_and_hypotenuse`
4. Answer schema: `integer`
5. Annotation schema: `point_map`
6. Scalar annotation checked: true

## Program Contract
- `solve_formula(right_triangle_leg_projection_relation, target=leg_or_projection_length, formula_schema=leg_geometric_mean_with_hypotenuse_projection); scene=triangle_relations; scope=leg_projection_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Semantics
- `leg_from_hypotenuse_projection` asks for a leg length from the full hypotenuse and adjacent projection.
- `projection_from_leg_and_hypotenuse` asks for one projection from the adjacent leg and full hypotenuse.

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `triangle_relations`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses keyed pixel points for the labeled construction points `A`, `B`, `C`, and `D`. Segment labels, right-angle markers, and target role remain visible diagram content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/triangle_relations.yaml`
- Task module: `src/trace_tasks/tasks/geometry/triangle_relations/leg_projection_length_value.py`
