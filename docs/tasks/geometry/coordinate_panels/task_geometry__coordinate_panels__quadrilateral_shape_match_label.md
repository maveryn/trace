# `task_geometry__coordinate_panels__quadrilateral_shape_match_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_panels`
5. Query id: `parallelogram_shape_match_label`, `rectangle_shape_match_label`, `rhombus_shape_match_label`, `square_shape_match_label`
6. Answer schema: `option_letter`
7. Annotation schema: `point_set`

## Program Contract
- `label(select_panel(candidate_coordinate_panels_6, quadrilateral_type)); scene=coordinate_panels; scope=quadrilateral_shape_match_label`

## Reasoning Operations

Families: `spatial_relations`, `matching`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_panels`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses the four pixel-space points plotted inside the selected panel. Panel boxes, graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_panels.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_panels/quadrilateral_shape_match_label.py`
