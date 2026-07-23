# `task_geometry__concentric_chord__chord_length_from_radii`

## Contract
1. Domain: `geometry`
2. Scene id: `concentric_chord`
3. Query id: `single`
4. Internal query id: `chord_length_from_radii`
5. Answer schema: `integer`
6. Annotation schema: `point_map` with keys `O`, `A`, `B`, `T`

## Program Contract
- `solve_formula(visible_concentric_chord_measurements, unknown_role=length_measure, formula_schema=chord_length_from_radii); scene=concentric_chord; scope=chord_length_from_radii`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `concentric_chord`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/concentric_chord.yaml`
- Task module: `src/trace_tasks/tasks/geometry/concentric_chord/chord_length_from_radii.py`
