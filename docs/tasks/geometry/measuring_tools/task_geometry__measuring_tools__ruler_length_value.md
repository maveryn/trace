# `task_geometry__measuring_tools__ruler_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `measuring_tools`
5. Supported `query_id`s: `single`
6. Answer schema: `integer`
7. Annotation schema: `segment`

## Program Contract
- `read_visible_measurement_tool(tool=ruler, candidate=marked_length_on_carrier_shape, unit=centimeters, operation=read_ruler_span, output_role=length_measure_integer, annotation_witness=measured_segment); scene=measuring_tools; scope=ruler_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `measuring_tools`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Annotation must be a
scalar segment `[[x0,y0],[x1,y1]]` marking the measured visual length.

## Internal Sampling
The carrier shape is internal construction metadata, not public query
variation. Supported carrier shapes are `circle`, `triangle`,
`parallelogram`, and `trapezoid`.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/measuring_tools.yaml`
- Task module: `src/trace_tasks/tasks/geometry/measuring_tools/ruler_length_value.py`
