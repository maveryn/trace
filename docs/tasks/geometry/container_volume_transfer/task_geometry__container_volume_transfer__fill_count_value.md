# `task_geometry__container_volume_transfer__fill_count_value`

## Contract
1. Domain: `geometry`
2. Scene id: `container_volume_transfer`
5. Query ids: `cone_to_cylinder_fill_count`, `cylinder_to_cuboid_fill_count`
6. Answer schema: `integer_value`
7. Annotation schema: `bbox_map`

## Program Contract
- `solve_formula(visible_source_target_container_transfer, unknown_role=full_pour_count, formula_schema=container_volume_transfer_fill_count); scene=container_volume_transfer; scope=fill_count_value`

## Reasoning Operations

Families: `counting`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_container_volume_transfer_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Annotation is a keyed bbox map with `source_container_bbox` and `target_container_bbox`. Numeric dimension labels, unit labels, transfer arrows, and the full-pours question mark remain visible in the image plus private verifier metadata, but are not requested as annotation.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/container_volume_transfer.yaml`
- Task module: `src/trace_tasks/tasks/geometry/container_volume_transfer/fill_count_value.py`
