# `task_geometry__function_panels__intersection_property_label`

## Contract
1. Domain: `geometry`
2. Scene id: `function_panels`
3. Public task id: `task_geometry__function_panels__intersection_property_label`
4. Supported `query_id`: `line_circle_tangent_label`, `line_circle_two_intersections_label`, `circle_circle_two_intersections_label`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`

## Program Contract
- `label(select_panel(candidate_coordinate_primitive_pairs, requested_intersection_condition)); scene=function_panels; scope=intersection_property_label`

## Reasoning Operations

Families: `spatial_relations`

## Prompt Bundle
- Prompt text is loaded from `geometry_analytical_intersection_property_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is one scalar pixel bbox around the selected panel. Visible intersection-point boxes remain private trace metadata, not prompt-facing annotation.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/function_panels.yaml`
- Task module: `src/trace_tasks/tasks/geometry/function_panels/intersection_property_label.py`
