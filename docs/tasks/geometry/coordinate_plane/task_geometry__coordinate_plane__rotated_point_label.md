# `task_geometry__coordinate_plane__rotated_point_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_plane`
5. Query id: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `point`

## Program Contract
- `label(select_candidate_point(candidate_points, coordinate_rule=rotation_about_marked_center, rotation_rule)); scene=coordinate_plane; scope=rotated_point_label`

## Reasoning Operations

Families: `transformation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_plane`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the scalar pixel point `[x,y]` at the center of the selected candidate rotated point. Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_plane.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_plane/rotated_point_label.py`
