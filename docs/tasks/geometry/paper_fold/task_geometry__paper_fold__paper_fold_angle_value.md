# `task_geometry__paper_fold__paper_fold_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `paper_fold`
3. Task id: `task_geometry__paper_fold__paper_fold_angle_value`
4. Query id: `single`
5. Answer schema: `number`
6. Answer precision: `one_decimal`
7. Annotation schema: `bbox_map`

## Program Contract
- `solve_formula(visible_paper_fold_angle_labels, unknown_role=half_angle_x, formula_schema=fold_bisector_with_straight_angle, output=angle_degrees_1dp); scene=paper_fold; scope=paper_fold_angle_value`

## Reasoning Operations

Families: `transformation`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `paper_fold`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. The task returns a `bbox_map` with keys:

- `target_angle_cue`: the visible box around the equal marked angle cues `x`.
- `given_angle_label`: the visible box around the given angle label adjacent to the folded edge.

Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Sampling
The generator samples a rounded angle answer uniformly from a broad finite support, then selects a valid folded-corner construction for that answer. The visible numeric angle is the straight-line supplement to the two equal fold angles, so the target is `(180 - shown_angle) / 2`. This avoids over-representing common height/offset ratios that collapse to the same rounded angle.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/paper_fold.yaml`
- Task module: `src/trace_tasks/tasks/geometry/paper_fold/paper_fold_angle_value.py`
