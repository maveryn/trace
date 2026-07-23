# `task_geometry__graph_paper__quadrilateral_type_count`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__quadrilateral_type_count`
4. Supported `query_id`: `square_count`, `non_square_rectangle_count`, `non_square_rhombus_count`, `slanted_parallelogram_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract
- `count_quadrilateral_class(query_id={square_count|non_square_rectangle_count|non_square_rhombus_count|slanted_parallelogram_count}, target_class={square|non_square_rectangle|non_square_rhombus|slanted_parallelogram}, output_role=count); scene=graph_paper; scope=quadrilateral_set`
- Matching uses standard quadrilateral predicates from the rendered vertices. Squares are counted regardless of rotation; `non_square_rectangle_count` and `non_square_rhombus_count` explicitly exclude squares; `slanted_parallelogram_count` counts non-rectangular parallelograms, including non-square rhombuses.
- Quadrilaterals use class-preserving integer graph-paper vertices with variation in scale, grid-preserving orientation, aspect ratio, and slant; repeated instances of the same class are not required to be congruent.
- Count objects are placed using their actual graph-unit bounds so independent quadrilaterals do not overlap.

## Reasoning Operations

Families: `filtering`, `counting`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the unordered set of bounding boxes for every matching quadrilateral.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
