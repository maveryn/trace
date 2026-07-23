# `task_physics__ray_optics__ray_target_hit_count`

## Summary
- Domain: `physics`
- Scene id: `ray_optics`
- Implementation source: `src/trace_tasks/tasks/physics/ray_optics/ray_target_hit_count.py`

## Program Contract

Program: `count(filter(target_points, intersects(hidden_ray_path, target_point))); scene=ray_optics; scope=ray_target_hit_count`

Candidate set: the visible mirror/ray geometry, ray path segments, target markers, and bounce/hit labels inside the `ray_target_hit_count` objective scope.
Operands: `hidden_ray_path` (semantic_role, allowed `trace_solved_ray_path`, source `program_schema_concrete`); `target_points` (semantic_role, allowed `visible_target_points`, source `program_schema_concrete`).
Operation: evaluate `count` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_count` schema; The answer value is the count of target points touched by the solved ray path.
Annotation witnesses: `point_set` witnesses from the finalized render. Annotation is an unordered set of final-image pixel points at the touched target centers. Annotation and answer must come from the same generated execution trace.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Query Branches

Supported `query_id`s: `single`

| Query id | Program schema |
| --- | --- |
| `single` | `count(filter(target_points, intersects(hidden_ray_path, target_point))); scene=ray_optics; scope=ray_target_hit_count` |

## Program Metadata
- Program signatures: `physics.ray_path_event_count`
- Parameter axes: `fixed_query`
- Arguments:
  - `hidden_ray_path`: semantic_role; allowed `trace_solved_ray_path`; source `program_schema_concrete`
  - `target_points`: semantic_role; allowed `visible_target_points`; source `program_schema_concrete`
- Argument metadata status: `curated`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is the count of target points touched by the solved ray path.

## Annotation Contract
- Annotation schema: `point_set`
- Generator `annotation_gt.type`: `point_set`
- Annotation is an unordered set of final-image pixel points at the touched target centers.
- Annotation and answer must come from the same generated execution trace.

## Prompt And Trace Requirements
- Prompt text must come from the `physics_ray_optics_v1` scene prompt bundle.
- The internal prompt branch is recorded as `target_hit_count` in trace metadata.
- Render randomness, sampled fonts/styles, physical layout, solved ray path, answer, and annotation payloads must be explicit in the instance trace.
