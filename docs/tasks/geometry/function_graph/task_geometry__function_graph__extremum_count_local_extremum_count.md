# `task_geometry__function_graph__extremum_count_local_extremum_count`

## Contract
1. Domain: `geometry`
2. Scene id: `function_graph`
5. Query ids: `minimum`, `maximum`
6. Answer schema: `integer_count`
7. Annotation schema: `point_set`

## Program Contract
- `count(filter(function_graph_feature_points, feature_type=local_extremum, extremum_kind=minimum|maximum)); scene=function_graph; scope=extremum_count_local_extremum_count`

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `function_graph`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is a pixel-space `point_set` containing every visible local minimum or maximum point requested by the query. Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/function_graph.yaml`
- Task module: `src/trace_tasks/tasks/geometry/function_graph/extremum_count_local_extremum_count.py`
