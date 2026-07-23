# `task_geometry__measuring_tools__protractor_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `measuring_tools`
5. Supported `query_id`s: `single`
6. Answer schema: `integer`
7. Annotation schema: `point_map`

## Program Contract
- `read_visible_measurement_tool(tool=protractor, candidate=marked_angle_on_carrier_shape, unit=degrees, operation=read_protractor_tick, output_role=angle_measure_integer, annotation_witness={angle_vertex,protractor_reading_tick}); scene=measuring_tools; scope=protractor_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `measuring_tools`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Annotation must be a
keyed point map with `angle_vertex` and `protractor_reading_tick`.

## Internal Sampling
The carrier shape is internal construction metadata, not public query
variation. Supported carrier shapes are `triangle` and `quadrilateral`.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/measuring_tools.yaml`
- Task module: `src/trace_tasks/tasks/geometry/measuring_tools/protractor_angle_value.py`
