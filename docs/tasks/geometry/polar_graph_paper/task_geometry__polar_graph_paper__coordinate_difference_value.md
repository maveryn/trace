# task_geometry__polar_graph_paper__coordinate_difference_value

## Contract
1. Domain: `geometry`
2. Scene id: `polar_graph_paper`
3. Query ids: `radius_difference_value`, `angle_difference_value`
4. Answer schema: `integer`
5. Annotation schema: `point_map`

## Program Contract
- `compare_two_polar_graph_components(candidates={P,Q}, component={radius|angle_degrees}, operation={absolute_radius_difference|smaller_angular_difference}, output_role=integer_difference_value, annotation_witness=point_map_P_Q); scene=polar_graph_paper; scope=coordinate_difference_value`

## Reasoning Operations

Families: `formula_evaluation`

## Task Summary
- Scene: `polar_graph_paper`
- Objective contract: `coordinate_difference_value`
- Public task id: `task_geometry__polar_graph_paper__coordinate_difference_value`
- Query ids: `radius_difference_value`, `angle_difference_value`

## Answer Schema
- `integer`
- The answer is the requested coordinate difference between points `P` and `Q`.

## Annotation Schema
- `point_map`
- The annotation has exactly the keys `P` and `Q`, each mapped to the pixel point of that plotted marker.

## Query Semantics
- `radius_difference_value`: read the two polar radii and return their absolute difference.
- `angle_difference_value`: read the two polar angles and return the smaller angular difference in degrees.

## Rendering Notes
- The scene draws polar graph paper with two labeled points `P` and `Q`.
- Both points lie on polar ring/spoke intersections by construction.
