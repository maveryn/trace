# `task_geometry__coordinate_composite__boundary_point_match_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_composite`
3. Query id: `line_circle_boundary_point`, `circle_polygon_boundary_point`, or `line_polygon_boundary_point`
4. Answer schema: `option_letter`
5. Annotation schema: `point`

## Program Contract
- `select(candidate_point where on_boundary(candidate_point, object_a) and on_boundary(candidate_point, object_b)); scene=coordinate_composite; scope=boundary_point_match_label`

## Reasoning Operations

Families: `logical_composition`, `spatial_relations`, `matching`

## Prompt Bundle
- Prompt text is loaded from `geometry_coordinate_composite_v0`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the pixel point `[x,y]` for the selected marked candidate point. The image shows four marked candidate points labeled `A` through `D`. Candidate labels, object outlines, graph-paper grid, and symbolic object definitions remain visible annotations plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_composite.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_composite/boundary_point_match_label.py`
