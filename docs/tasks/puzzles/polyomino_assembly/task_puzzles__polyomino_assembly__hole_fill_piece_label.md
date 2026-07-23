# `task_puzzles__polyomino_assembly__hole_fill_piece_label`

## Contract

1. Domain: `puzzles`
2. Scene package: `src/trace_tasks/tasks/puzzles/polyomino_assembly/`
3. Scene id: `polyomino_assembly`
4. Public task id: `task_puzzles__polyomino_assembly__hole_fill_piece_label`
5. Supported `query_id` values: `single`
6. Prompt query key: `hole_fill_piece_label`
7. Answer schema: `option_letter`
8. Annotation schema: `bbox`
9. Program schema: `select_label(filler_piece_option, rule=option_matches_hole_under_rotation_or_reflection); scene=polyomino_assembly; scope=hole_fill_piece`

## Program Contract

Program: `select_label(filler_piece_option, rule=option_matches_hole_under_rotation_or_reflection); scene=polyomino_assembly; scope=hole_fill_piece`

Candidate set: the visible polyomino cells, source/target/hole shapes, allowed transforms, and labeled candidate options inside the `hole_fill_piece` objective scope.
Operands: visible scene state and prompt-bound operands named by `filler_piece_option`, `option_matches_hole_under_rotation_or_reflection`, `polyomino_assembly`, `hole_fill_piece`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the selected option label.
Annotation witnesses: `annotation` uses the `bbox` schema; one image-pixel bbox around the selected option card.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Contract

- Supported public `query_id`: `single`
- The task shows one polyomino board with a blank interior hole and four labeled candidate filler pieces.
- Hole shape, option order, scene treatment, font, and theme are generation/render axes, not public taxonomy axes.

## Generation Contract

- The hole shape is a connected polyomino placed inside a filled rectangular polyomino board.
- The hole does not touch the outside boundary of the board.
- Exactly one option piece matches the hole under rotations or reflections.
- Distractors have the same cell count as the hole and are verified against the reflection-aware shape signature.
- Options are always labeled `A` through `D`.

## Prompt Contract

- Bundle: `puzzles_polyomino_assembly_v1`
- `scene_key`: `polyomino_assembly`
- `task_key`: `hole_fill_piece_label_query`
- `query_key`: `hole_fill_piece_label`
- Prompt-facing answer is the selected option label.
- Prompt-facing annotation is one image-pixel bbox around the selected option card.
- Prompt wording must state that rotations and flips are allowed.

## Annotation + Trace Contract

- `answer_gt.type`: `option_letter`
- `annotation_gt.type`: `bbox`
- `projected_annotation` includes `bbox` and `pixel_bbox`.
- `render_map.item_bboxes_px` stores option-card bboxes keyed by option choice id.
- `execution_trace` records public query `single`, transform policy, option specs, board cells, hole cells, answer label, and solver trace.
- Answer and annotation are projected from the same selected option card id.
- `scalar_annotation_checked=true`.

## Determinism

- Deterministic sampling/rendering from `instance_seed`, scene config, prompt bundle, and code version.
- No semantic auto-relaxation is used to force acceptance.
