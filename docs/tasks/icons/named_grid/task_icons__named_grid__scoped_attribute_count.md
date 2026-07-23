# `task_icons__named_grid__scoped_attribute_count`

- domain: `icons`
- scene_id: `named_grid`
- task: `scoped_attribute_count`
- module: `src/trace_tasks/tasks/icons/named_grid/scoped_attribute_count.py`

## Program Contract

Program: `count.scoped_attribute(scene=named_grid, scope=numbered_row_or_column, attribute=shape, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `numbered_row_or_column` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_grid`, `numbered_row_or_column`, `attribute`, `shape`.
Operation: evaluate `count.scoped_attribute` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Annotation schema: `bbox_set`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Query IDs
- `row_shape_count`
- `column_shape_count`

## Generation
Default answer support is `1..5`. Grid sizes are sampled from every
row/column combination in `4..6`:

- `4 x 4`
- `4 x 5`
- `4 x 6`
- `5 x 4`
- `5 x 5`
- `5 x 6`
- `6 x 4`
- `6 x 5`
- `6 x 6`

The selected grid size must have enough cells in the queried row or column to
support the sampled answer. The queried line contains exactly the sampled
answer count of the target shape. Additional target-shape icons are placed
outside the queried row/column as distractors so the task requires row/column
localization rather than whole-image counting.

Fill style and color are rendered as non-semantic visual variation.

## Trace
The trace records:
- every named icon with row/column indices, row/column numbers, cell bbox, and
  icon bbox,
- the target shape id/name,
- the queried axis and one-based row/column number,
- the counted cells and off-line target distractor cells,
- render-style metadata for the panel title and row/column number-label text
  legibility,
- query, answer, grid-size, line-index, shape, and fill-style probability
  metadata.
