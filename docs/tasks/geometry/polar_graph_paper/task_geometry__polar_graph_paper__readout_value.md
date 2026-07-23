# task_geometry__polar_graph_paper__readout_value

## Contract
1. Domain: `geometry`
2. Scene id: `polar_graph_paper`
3. Query ids: `radius_readout_value`, `angle_readout_value`
4. Answer schema: `integer`
5. Annotation schema: `point`

## Program Contract
- `read_polar_graph_component(candidate=P, component={radius|angle_degrees}, operation=read_ring_or_spoke_value, output_role=integer_component_value, annotation_witness=point_P); scene=polar_graph_paper; scope=readout_value`

## Reasoning Operations

Families: `direct_retrieval`

## Task Summary
- Scene: `polar_graph_paper`
- Objective contract: `readout_value`
- Public task id: `task_geometry__polar_graph_paper__readout_value`
- Query ids: `radius_readout_value`, `angle_readout_value`

## Answer Schema
- `integer`
- The answer is the requested radius or angle value read from the polar graph.

## Annotation Schema
- `point`
- The annotation is the pixel point at plotted point `P`.

## Query Semantics
- `radius_readout_value`: read the polar radius of point `P`.
- `angle_readout_value`: read the polar angle of point `P` in degrees.

## Rendering Notes
- The scene draws polar graph paper and point `P`.
- The plotted point lies on a polar ring/spoke intersection by construction.
