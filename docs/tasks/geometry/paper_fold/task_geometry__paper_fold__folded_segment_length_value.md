# `task_geometry__paper_fold__folded_segment_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `paper_fold`
3. Task id: `task_geometry__paper_fold__folded_segment_length_value`
4. Query id: `single`
5. Answer schema: `integer`
6. Annotation schema: `segment`

## Program Contract
- `solve_formula(visible_paper_fold_side_labels, unknown_role=folded_segment_length, formula_schema=pythagorean_leg_then_fold_correspondence, output=integer_length); scene=paper_fold; scope=folded_segment_length_value`

## Reasoning Operations

Families: `transformation`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `paper_fold`.
- Prompt modes: `answer_only` and `answer_and_annotation`.
- The task prompt states that triangle `AEF` is folded along crease `EF` so
  `A` lands on `P`; this makes the side correspondence `AE = PE` and `AF = PF`
  visually grounded instead of implicit.

## Annotation
Prompt-facing annotation is the requested folded segment only: `[[x0,y0],[x1,y1]]`. For example, if the prompt asks for `FP`, the segment endpoints are the visible points `F` and `P`.

Numeric labels, point labels, formulas, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Sampling
The generator samples an integer target answer uniformly from a broad finite support built from integer right-triangle triples, then selects a compatible fold construction for that answer. One original leg and the crease length are visible; the target folded segment is equal to the missing original leg after folding.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/paper_fold.yaml`
- Task module: `src/trace_tasks/tasks/geometry/paper_fold/folded_segment_length_value.py`
