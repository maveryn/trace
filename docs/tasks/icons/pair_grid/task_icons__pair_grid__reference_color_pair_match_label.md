# `task_icons__pair_grid__reference_color_pair_match_label`

## Identity
- domain: `icons`
- scene_id: `pair_grid`
- module: `src/trace_tasks/tasks/icons/pair_grid/reference_color_pair_match_label.py`
- prompt bundle: `icons_pair_grid_v1`

## Program Contract

Program: `selection.reference_relation_match(scene=pair_grid, scope=labeled_scene_cells, reference_relation=left_right_color_pair, output=option_letter)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `labeled_scene_cells` objective scope.
Operands: visible scene state and prompt-bound operands named by `pair_grid`, `labeled_scene_cells`, `reference_relation`, `left_right_color_pair`.
Operation: evaluate `selection.reference_relation_match` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Contract
Renders a Reference before/after icon pair and a labeled Scene grid of six
before/after icon-pair cells, then asks which labeled Scene cell has the same
left and right colors as the Reference pair. Exactly one Scene cell is correct
by construction.

Supported query ids:
- `single`

Answer schema: `option_letter`.
Annotation schema: scalar `bbox` around the selected Scene cell.

## Notes
This task holds size and transform constant while sampling different icon shapes
for the Reference and Scene cells. Shape is non-answer-bearing; the only
answer-bearing relation is the left/right color pair. Distractors include
same-left/different-right, different-left/same-right, reversed-color, and
both-different color-pair cases when the sampled palette permits them.

Renderer metadata records sampled palette/style, panel-header and cell-label
text-legibility metadata, and per-icon noise edits.
