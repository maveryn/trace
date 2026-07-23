# `task_puzzles__polyomino_assembly__decomposition_pair_label`

## Contract

1. Domain: `puzzles`
2. Scene package: `src/trace_tasks/tasks/puzzles/polyomino_assembly/`
3. Scene id: `polyomino_assembly`
4. Public task id: `task_puzzles__polyomino_assembly__decomposition_pair_label`
5. Supported `query_id` values: `single`
6. Prompt query key: `decomposition_pair_label`
7. Answer schema: `option_letter`
8. Annotation schema: `bbox`
9. Program schema: `select_label(option_pair, rule=two_rotatable_pieces_tile_target_exactly); scene=polyomino_assembly; scope=decomposition_pair`

## Program Contract

Program: `select_label(option_pair, rule=two_rotatable_pieces_tile_target_exactly); scene=polyomino_assembly; scope=decomposition_pair`

Candidate set: the visible polyomino cells, source/target/hole shapes, allowed transforms, and labeled candidate options inside the `decomposition_pair` objective scope.
Operands: visible scene state and prompt-bound operands named by `option_pair`, `two_rotatable_pieces_tile_target_exactly`, `polyomino_assembly`, `decomposition_pair`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the selected option label.
Annotation witnesses: `annotation` uses the `bbox` schema; one image-pixel bbox around the selected option card.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Contract

- Supported public `query_id`: `single`
- The task shows one target polyomino and four labeled two-piece options.
- Piece shapes, target size, option order, scene treatment, font, and theme are generation/render axes, not public taxonomy axes.

## Generation Contract

- The target shape is a connected polyomino.
- Exactly one option pair can tile the target shape with no gaps or overlaps.
- Pieces may be translated and rotated; pieces may not be reflected/flipped.
- Distractors have the same total cell count as the target and are verified against the same solver.
- Options are always labeled `A` through `D`.

## Prompt Contract

- Bundle: `puzzles_polyomino_assembly_v1`
- `scene_key`: `polyomino_assembly`
- `task_key`: `decomposition_pair_label_query`
- `query_key`: `decomposition_pair_label`
- Prompt-facing answer is the selected option label.
- Prompt-facing annotation is one image-pixel bbox around the selected option card.

## Annotation + Trace Contract

- `answer_gt.type`: `option_letter`
- `annotation_gt.type`: `bbox`
- `projected_annotation` includes `bbox` and `pixel_bbox`.
- `render_map.item_bboxes_px` stores option-card bboxes keyed by option choice id.
- `execution_trace` records public query `single`, transform policy, option specs, target cells, answer label, and solver trace.
- Answer and annotation are projected from the same selected option card id.
- `scalar_annotation_checked=true`.

## Determinism

- Deterministic sampling/rendering from `instance_seed`, scene config, prompt bundle, and code version.
- No semantic auto-relaxation is used to force acceptance.
