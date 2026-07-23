# `task_geometry__area_partition__total_area_value`

## Contract
1. Domain: `geometry`
2. Scene id: `area_partition`
4. Query id: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `bbox_map`

## Program Contract
- `solve_formula(visible_area_partition_measurements, unknown_role=area_measure, formula_schema=shaded_unit_fraction_area_to_total_area, partition_rule=visible_fraction_partition_rule); scene=area_partition; scope=total_area_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `area_partition`.
- Prompt schema: external prompt bundle
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/area_partition.yaml`
- Task module: `src/trace_tasks/tasks/geometry/area_partition/total_area_value.py`
