# `task_geometry__graph_paper__perimeter_extremum_label`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__perimeter_extremum_label`
4. Supported `query_id`: `largest`, `smallest`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`

## Program Contract
- `select_shape_perimeter_extremum(extremum={largest|smallest}, shape_family={rectangle|right_triangle}, output_role=shape_label); scene=graph_paper; scope=labeled_shape_set`
- Labeled shapes use integer graph-paper vertices and are placed with non-overlapping graph-unit bounding boxes.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the selected shape bounding box in pixel coordinates.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
