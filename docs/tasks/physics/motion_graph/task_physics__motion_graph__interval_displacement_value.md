# `task_physics__motion_graph__interval_displacement_value`

## Summary
- Domain: `physics`
- Scene id: `motion_graph`
- Implementation scene: `motion_graph`
- Implementation source: `src/trace_tasks/tasks/physics/motion_graph/interval_displacement_value.py`

## Task Contract
Computes the integer displacement over a marked interval on a velocity-time graph.

## Program Contract

Program: `integer(area_under_velocity_time_curve(marked_interval, segment_mode=constant_velocity_or_constant_acceleration)); scene=motion_graph; scope=interval_displacement_value`

Candidate set: the visible graph curve, marked interval, axis scales, labels, and option cells when present inside the `interval_displacement_value` objective scope.
Operands: `marked_interval` (semantic_role, allowed `highlighted_time_interval_on_velocity_time_graph`, source `program_schema_concrete`); `velocity_segment` (semantic_role, allowed `constant_or_linearly_changing_velocity_segment`, source `program_schema_concrete`); `axis_scale` (semantic_role, allowed `visible_integer_time_and_velocity_axis_scale`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the displacement in meters over the marked interval.
Annotation witnesses: `segment` witnesses from the finalized render. Annotation value: `[[x0, y0], [x1, y1]]` for the marked velocity-time graph segment, where each endpoint is a `[x, y]` pixel point. Annotation must mark the minimal visual segment witness for the marked interval. It must not mark derived displacement text, prompt-only formulas, decorative panel chrome, broad axis-scale regions, or unrelated graph segments.
Query ids: `constant_velocity_interval_displacement`, `constant_acceleration_interval_displacement`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `constant_velocity_interval_displacement` | `integer(area_under_velocity_time_curve(marked_interval, segment_mode=constant_velocity)); scene=motion_graph; scope=interval_displacement_value` |
| `constant_acceleration_interval_displacement` | `integer(area_under_velocity_time_curve(marked_interval, segment_mode=constant_acceleration)); scene=motion_graph; scope=interval_displacement_value` |

## Program Metadata
- Program signatures: `physics.motion_graph_interval_displacement`
- Base program contract: `integer(area_under_velocity_time_curve(marked_interval, segment_mode=constant_velocity_or_constant_acceleration)); scene=motion_graph; scope=interval_displacement_value`
- Parameter axes: `query_id`, `scene_variant`, `interval_width`, `velocity_endpoints`
- Arguments:
  - `marked_interval`: semantic_role; allowed `highlighted_time_interval_on_velocity_time_graph`; source `program_schema_concrete`
  - `velocity_segment`: semantic_role; allowed `constant_or_linearly_changing_velocity_segment`; source `program_schema_concrete`
  - `axis_scale`: semantic_role; allowed `visible_integer_time_and_velocity_axis_scale`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id`s: `constant_velocity_interval_displacement`, `constant_acceleration_interval_displacement`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the displacement in meters over the marked interval.

## Annotation Contract
- Annotation schema: `segment`
- Generator `annotation_gt.type`: `segment`
- Annotation value: `[[x0, y0], [x1, y1]]` for the marked velocity-time graph segment, where each endpoint is a `[x, y]` pixel point.
- Annotation must mark the minimal visual segment witness for the marked interval. It must not mark derived displacement text, prompt-only formulas, decorative panel chrome, broad axis-scale regions, or unrelated graph segments.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics motion-graph v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Constant-velocity intervals use rectangle area `v * delta_t`; constant-acceleration intervals use trapezoid area `((v_start + v_end) / 2) * delta_t`.
- Endpoint values and interval widths must be sampled so the answer is integer-valued.
- Render randomness, sampled fonts/styles, marked interval, curve values, and verifier payloads must be explicit in the instance trace.
