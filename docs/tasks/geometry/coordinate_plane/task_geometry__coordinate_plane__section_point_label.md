# `task_geometry__coordinate_plane__section_point_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_plane`
5. Query id: `one_third_from_p_to_q`, `two_thirds_from_p_to_q`
6. Answer schema: `option_letter`
7. Annotation schema: `point`

## Program Contract
- `label(select_candidate_point(candidate_points, coordinate_rule=section_formula, section_ratio=ratio_from_p_to_q)); scene=coordinate_plane; scope=section_point_label`
- The task renders four lettered candidate points. All candidates are integer lattice points strictly on segment `PQ`; off-segment distractors are not allowed.

## Reasoning Operations

Families: `spatial_relations`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_plane`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the scalar pixel point `[x,y]` at the center of the selected candidate section point. Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_plane.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_plane/section_point_label.py`
