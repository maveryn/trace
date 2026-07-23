# `task_physics__ray_optics__ray_bounce_count`

## Summary
- Domain: `physics`
- Scene id: `ray_optics`
- Implementation source: `src/trace_tasks/tasks/physics/ray_optics/ray_bounce_count.py`

## Program Contract

Program: `count(reflection_points(hidden_ray_path)); scene=ray_optics; scope=ray_bounce_count`

Candidate set: the visible mirror/ray geometry, ray path segments, target markers, and bounce/hit labels inside the `ray_bounce_count` objective scope.
Operands: `hidden_ray_path` (semantic_role, allowed `trace_solved_ray_path`, source `program_schema_concrete`).
Operation: evaluate `count` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_count` schema; The answer value is the count of mirror-bounce points on the solved ray path.
Annotation witnesses: `point_set` witnesses from the finalized render. Annotation is an unordered set of final-image pixel points at the mirror-bounce centers. Annotation and answer must come from the same generated execution trace.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `spatial_relations`, `transformation`

## Query Branches

Supported `query_id`s: `single`

| Query id | Program schema |
| --- | --- |
| `single` | `count(reflection_points(hidden_ray_path)); scene=ray_optics; scope=ray_bounce_count` |

## Program Metadata
- Program signatures: `physics.ray_path_event_count`
- Parameter axes: `fixed_query`
- Arguments:
  - `hidden_ray_path`: semantic_role; allowed `trace_solved_ray_path`; source `program_schema_concrete`
- Argument metadata status: `curated`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is the count of mirror-bounce points on the solved ray path.

## Annotation Contract
- Annotation schema: `point_set`
- Generator `annotation_gt.type`: `point_set`
- Annotation is an unordered set of final-image pixel points at the mirror-bounce centers.
- Annotation and answer must come from the same generated execution trace.

## Prompt And Trace Requirements
- Prompt text must come from the `physics_ray_optics_v1` scene prompt bundle.
- The internal prompt branch is recorded as `bounce_count` in trace metadata.
- Render randomness, sampled fonts/styles, physical layout, solved ray path, answer, and annotation payloads must be explicit in the instance trace.
