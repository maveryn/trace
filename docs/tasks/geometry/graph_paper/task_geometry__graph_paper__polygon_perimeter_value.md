# `task_geometry__graph_paper__polygon_perimeter_value`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__polygon_perimeter_value`
4. Supported `query_id`: `single`
5. Answer schema: `integer`
6. Annotation schema: `point_set`

## Program Contract
- `compute_lattice_polygon_perimeter(target=polygon, shape_family={rectangle|triangle|parallelogram}, output_role=perimeter_integer); scene=graph_paper; scope=single_polygon`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the unordered set of polygon vertex pixel points.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
