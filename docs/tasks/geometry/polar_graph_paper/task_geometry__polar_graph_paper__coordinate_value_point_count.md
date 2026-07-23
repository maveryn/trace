# task_geometry__polar_graph_paper__coordinate_value_point_count

## Contract
1. Domain: `geometry`
2. Scene id: `polar_graph_paper`
3. Query ids: `radius_value_point_count`, `angle_value_point_count`
4. Answer schema: `integer`
5. Annotation schema: `point_set`

## Program Contract
- `count(marked_points where polar_coordinate_component(point, component={radius|angle_degrees}) equals prompted_value); scene=polar_graph_paper; scope=coordinate_value_point_count`

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `formula_evaluation`

## Task Summary
- Scene: `polar_graph_paper`
- Objective contract: `coordinate_value_point_count`
- Public task id: `task_geometry__polar_graph_paper__coordinate_value_point_count`
- Query ids: `radius_value_point_count`, `angle_value_point_count`

## Answer Schema
- `integer`
- The answer is the number of marked points matching the prompted radius or angle value.
- The generated answer support is `1..5`, with `8..12` marked points shown.

## Annotation Schema
- `point_set`
- The annotation is the unordered set of pixel points for all marked points counted in the answer.

## Query Semantics
- `radius_value_point_count`: count marked points whose polar radius equals the prompted ring value.
- `angle_value_point_count`: count marked points whose polar angle equals the prompted spoke value in degrees.

## Rendering Notes
- The scene draws polar graph paper with marked point markers.
- Every marked point lies on a polar ring/spoke intersection by construction.
- Matching and distractor points use the same visual style; the answer is determined by polar coordinate values, not marker color.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/polar_graph_paper.yaml`
- Task module: `src/trace_tasks/tasks/geometry/polar_graph_paper/coordinate_value_point_count.py`
