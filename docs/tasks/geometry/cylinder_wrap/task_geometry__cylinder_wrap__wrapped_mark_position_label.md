# `task_geometry__cylinder_wrap__wrapped_mark_position_label`

## Contract
1. Domain: `geometry`
2. Scene id: `cylinder_wrap`
5. Query id: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `point_map`

## Program Contract
- `label(select_rim_candidate(rim_positions, unwrap_mapping(strip_mark)), output_role=matching_candidate_label); scene=cylinder_wrap; scope=wrapped_mark_position_label`

## Reasoning Operations

Families: `transformation`, `matching`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `cylinder_wrap`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

Annotation contract note: this task intentionally stays `point_map` because it binds two role-specific point witnesses: the source strip mark and the matching rim candidate.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/cylinder_wrap.yaml`
- Task module: `src/trace_tasks/tasks/geometry/cylinder_wrap/wrapped_mark_position_label.py`
