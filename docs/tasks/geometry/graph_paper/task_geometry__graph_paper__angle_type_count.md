# `task_geometry__graph_paper__angle_type_count`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__angle_type_count`
4. Supported `query_id`: `acute_angle_count`, `right_angle_count`, `obtuse_angle_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract
- `count_angle_class(query_id={acute_angle_count|right_angle_count|obtuse_angle_count}, target_class={acute|right|obtuse}, output_role=count); scene=graph_paper; scope=angle_set`
- Count objects are placed using their actual graph-unit bounds so independent angle drawings do not overlap.

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the unordered set of bounding boxes for every matching angle.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
