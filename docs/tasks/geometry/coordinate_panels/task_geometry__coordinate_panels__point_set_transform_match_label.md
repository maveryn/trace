# `task_geometry__coordinate_panels__point_set_transform_match_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_panels`
5. Query id: `reflection_x_match_label`, `reflection_y_match_label`, `rotation_180_match_label`, `translation_match_label`
6. Answer schema: `option_letter`
7. Annotation schema: `point_set`

## Program Contract
- `label(select_panel(candidate_coordinate_panels_6, point_set_transform)); scene=coordinate_panels; scope=point_set_transform_match_label`

## Reasoning Operations

Families: `transformation`, `matching`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_panels`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the pixel-space source points followed by the candidate points in the selected panel. Panel boxes, graph coordinates, transform flags, and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_panels.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_panels/point_set_transform_match_label.py`
