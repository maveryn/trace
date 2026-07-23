# `task_geometry__incircle_tangents__incircle_radius_from_area_value`

## Contract
1. Domain: `geometry`
2. Scene id: `incircle_tangents`
3. Query id: `single`
4. Internal query id: `inradius_from_tangent_segments`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `point_map` with keys `A`, `B`, `C`, `D`, `E`, `F`, `O`

## Program Contract
- `radius(incircle_tangent_triangle, tangent_equalities={AD=AF,BD=BE,CE=CF}); scene=incircle_tangents; scope=incircle_radius_from_area_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_tangent_polygon_incircle_v0`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space keyed points for the triangle vertices, tangent contact points, and incircle center. Equality of each tangent pair is marked visually by matching tick marks. The incircle radius is computed from the tangent lengths using the tangent-triangle relation; formula metadata, case ids, and unrendered measurements remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/incircle_tangents.yaml`
- Task module: `src/trace_tasks/tasks/geometry/incircle_tangents/incircle_radius_from_area_value.py`
