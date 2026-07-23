# `task_geometry__circle_pair_tangents__external_tangent_segment_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_pair_tangents`
4. Query ids: `tangent_segment_length_from_center_distance`, `center_distance_from_tangent_segment_length`
5. Answer schema: `integer_value`
6. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_external_common_tangent_measurements, unknown_role=tangent_length|center_distance, formula_schema=external_common_tangent_right_triangle); scene=circle_pair_tangents; scope=external_tangent_segment_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_circle_pair_tangents_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Annotation is a keyed point map with visible construction-point roles `C`, `D`, `A`, and `B`, marking the two circle centers and two tangent points. The auxiliary point `E`, segment labels, right-angle markers, numeric labels, and the symbolic tangent relation remain visible context plus private verifier metadata.

Scalar annotation does not apply because the task always needs multiple role-bound construction points.

## Rendering
The diagram explicitly draws the auxiliary right triangle `C-E-D`: `CE` is parallel to the common tangent `AB`, `ED` is labeled with the radius difference, and `CD` is the center-to-center distance. This makes the external-tangent right-triangle relation visible without changing the answer or annotation contract.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_pair_tangents.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_pair_tangents/external_tangent_segment_length_value.py`
