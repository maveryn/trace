# `task_puzzles__polyomino_assembly__composition_result_label`

## Contract

1. Domain: `puzzles`
2. Scene package: `src/trace_tasks/tasks/puzzles/polyomino_assembly/`
3. Scene id: `polyomino_assembly`
4. Public task id: `task_puzzles__polyomino_assembly__composition_result_label`
5. Supported `query_id` values: `single`
6. Prompt query key: `composition_result_label`
7. Answer schema: `option_letter`
8. Annotation schema: `bbox`
9. Program schema: `select_label(composite_shape_option, rule=two_source_pieces_tile_option_exactly); scene=polyomino_assembly; scope=composition_result`

## Program Contract

Program: `select_label(composite_shape_option, rule=two_source_pieces_tile_option_exactly); scene=polyomino_assembly; scope=composition_result`

Candidate set: the visible polyomino cells, source/target/hole shapes, allowed transforms, and labeled candidate options inside the `composition_result` objective scope.
Operands: visible scene state and prompt-bound operands named by `composite_shape_option`, `two_source_pieces_tile_option_exactly`, `polyomino_assembly`, `composition_result`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the selected option label.
Annotation witnesses: `annotation` uses the `bbox` schema; one image-pixel bbox around the selected option card.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Contract

- Supported public `query_id`: `single`
- The task shows two source polyomino pieces and four labeled candidate result shapes.
- Source piece shape, result size, option order, scene treatment, font, and theme are generation/render axes, not public taxonomy axes.

## Generation Contract

- The two source pieces are connected polyominoes.
- Exactly one option shape can be tiled by the two source pieces with no gaps or overlaps.
- Pieces may be translated and rotated; pieces may not be reflected/flipped.
- Distractors have the same total cell count as the source pieces and are verified against the same solver.
- Options are always labeled `A` through `D`.

## Prompt Contract

- Bundle: `puzzles_polyomino_assembly_v1`
- `scene_key`: `polyomino_assembly`
- `task_key`: `composition_result_label_query`
- `query_key`: `composition_result_label`
- Prompt-facing answer is the selected option label.
- Prompt-facing annotation is one image-pixel bbox around the selected option card.

## Annotation + Trace Contract

- `answer_gt.type`: `option_letter`
- `annotation_gt.type`: `bbox`
- `projected_annotation` includes `bbox` and `pixel_bbox`.
- `render_map.item_bboxes_px` stores option-card bboxes keyed by option choice id.
- `execution_trace` records public query `single`, transform policy, option specs, source piece cells, answer label, and solver trace.
- Answer and annotation are projected from the same selected option card id.
- `scalar_annotation_checked=true`.

## Determinism

- Deterministic sampling/rendering from `instance_seed`, scene config, prompt bundle, and code version.
- No semantic auto-relaxation is used to force acceptance.
