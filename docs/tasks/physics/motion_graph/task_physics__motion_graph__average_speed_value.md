# `task_physics__motion_graph__average_speed_value`

## Summary
- Domain: `physics`
- Scene id: `motion_graph`
- Implementation scene: `motion_graph`
- Implementation source: `src/trace_tasks/tasks/physics/motion_graph/average_speed_value.py`

## Task Contract
Computes the integer average speed over a marked interval on a distance-time graph.

## Program Contract

Program: `integer((d_end - d_start) / (t_end - t_start)); scene=motion_graph; scope=average_speed_value`

Candidate set: the visible graph curve, marked interval, axis scales, labels, and option cells when present inside the `average_speed_value` objective scope.
Operands: `marked_distance_time_graph_interval` (semantic_role, allowed `highlighted_time_interval_with_local_distance_curve_segment`, source `program_schema_concrete`); `axis_scale` (query_operand, allowed `visible_time_and_distance_axis_scale`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the average speed in `m/s` over the marked interval.
Annotation witnesses: `segment` witnesses from the finalized render. Annotation value: `[[x0, y0], [x1, y1]]` for the marked distance-time graph segment, where each endpoint is a `[x, y]` pixel point. Annotation must mark the minimal visual segment witness for the marked interval. It must not mark option boxes, decorative graph chrome, broad axis-scale regions, or derived answer text.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(average_speed(marked_distance_time_graph_interval)); scene=motion_graph; scope=average_speed_value` |

## Program Metadata
- Program signatures: `physics.motion_graph_average_speed_value`
- Base program contract: `integer((d_end - d_start) / (t_end - t_start)); scene=motion_graph; scope=average_speed_value`
- Parameter axes: `scene_variant`, `average_speed_m_s`, `interval_width`
- Arguments:
  - `marked_distance_time_graph_interval`: semantic_role; allowed `highlighted_time_interval_with_local_distance_curve_segment`; source `program_schema_concrete`
  - `axis_scale`: query_operand; allowed `visible_time_and_distance_axis_scale`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id`s: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the average speed in `m/s` over the marked interval.

## Annotation Contract
- Annotation schema: `segment`
- Generator `annotation_gt.type`: `segment`
- Annotation value: `[[x0, y0], [x1, y1]]` for the marked distance-time graph segment, where each endpoint is a `[x, y]` pixel point.
- Annotation must mark the minimal visual segment witness for the marked interval. It must not mark option boxes, decorative graph chrome, broad axis-scale regions, or derived answer text.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics motion-graph v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- The public `query_id` is `single`.
- Render randomness, sampled fonts/styles, marked interval, distance values, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep axis labels, the marked interval, local segment, and scale readable.
