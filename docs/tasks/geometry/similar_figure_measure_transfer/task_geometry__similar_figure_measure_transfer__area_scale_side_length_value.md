# `task_geometry__similar_figure_measure_transfer__area_scale_side_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `similar_figure_measure_transfer`
5. Query id: `side_length_from_area_pair` or `side_length_from_area_ratio`
6. Answer schema: `integer`
7. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_similar_figure_area_scale_relation, unknown_role=target_corresponding_side_length, formula_schema=area_scale_to_linear_scale); scene=similar_figure_measure_transfer; scope=area_scale_side_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_similar_figure_measure_transfer_v0`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Query and Construction Axes
- `side_length_from_area_pair` uses two visible area labels.
- `side_length_from_area_ratio` uses a visible area-ratio label.
- `construction_family` is replay metadata for the area-pair branch and may choose a side-by-side or nested construction.

## Annotation
Prompt-facing annotation uses a `point_map` keyed by visible point labels for the corresponding side endpoints used to transfer the area scale.

Area labels, area ratios, tick marks, and derived scale factors remain visible scene content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/similar_figure_measure_transfer.yaml`
- Task module: `src/trace_tasks/tasks/geometry/similar_figure_measure_transfer/area_scale_side_length_value.py`
