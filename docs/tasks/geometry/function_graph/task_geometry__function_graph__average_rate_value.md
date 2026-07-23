# `task_geometry__function_graph__average_rate_value`

## Contract
1. Domain: `geometry`
2. Scene id: `function_graph`
3. Query ids: `single`
4. Answer schema: `number`
5. Answer precision: `one_decimal`
6. Annotation schema: `point_map` with keys `A` and `B`

## Program Contract
- `average_rate(marked_function_points, point_roles=A|B); scene=function_graph; scope=average_rate_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `function_graph`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is a pixel-space `point_map` binding the marked graph points `A` and `B`. Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/function_graph.yaml`
- Task module: `src/trace_tasks/tasks/geometry/function_graph/average_rate_value.py`
