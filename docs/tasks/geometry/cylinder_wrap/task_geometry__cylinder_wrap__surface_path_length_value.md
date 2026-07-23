# `task_geometry__cylinder_wrap__surface_path_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `cylinder_wrap`
5. Query id: `single`
6. Answer schema: `integer`
7. Annotation schema: `bbox_map`

## Program Contract
- `solve_formula(visible_cylinder_wrap_measurements, unknown_role=marked_surface_path_length, formula_schema=surface_path_pythagorean_length); scene=cylinder_wrap; scope=surface_path_length_value`

## Reasoning Operations

Families: `topology`, `transformation`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `cylinder_wrap`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

Annotation contract note: this task intentionally stays `bbox_map` because it binds three non-homogeneous visual roles: the marked path, circumference dimension, and height dimension.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/cylinder_wrap.yaml`
- Task module: `src/trace_tasks/tasks/geometry/cylinder_wrap/surface_path_length_value.py`
