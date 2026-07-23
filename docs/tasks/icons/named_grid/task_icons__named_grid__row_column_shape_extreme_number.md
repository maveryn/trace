# `task_icons__named_grid__row_column_shape_extreme_number`

- domain: `icons`
- scene_id: `named_grid`
- task: `row_column_shape_extreme_number`
- module: `src/trace_tasks/tasks/icons/named_grid/row_column_shape_extreme_number.py`

## Program Contract

Program: `selection.extreme_metric_label(scene=named_grid, scope=numbered_rows_or_columns, metric=target_shape_count, extrema=most|fewest, output=one_based_line_number)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `numbered_rows_or_columns` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_grid`, `numbered_rows_or_columns`, `metric`, `target_shape_count`, `extrema`, `most`, `fewest`, `one_based_line_number`.
Operation: evaluate `selection.extreme_metric_label` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `one_based_line_number` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; non-empty.
Annotation schema: `bbox_set`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`

## Query IDs
- `row_most_shape_number`
- `row_fewest_shape_number`
- `column_most_shape_number`
- `column_fewest_shape_number`

## Generation
Default answer-line support is `1..6`. Grid sizes are sampled from every
row/column combination in `4..6`.

The selected row or column has a unique target-shape count extreme by
construction. For most-count queries, every other row or column has fewer
target-shape icons. For fewest-count queries, every other row or column has
more target-shape icons. The winning line contains at least one target icon so
prompt-facing annotation is non-empty.

Fill style and color are rendered as non-semantic visual variation.

## Trace
The trace records:
- every named icon with row/column indices, row/column numbers, cell bbox, and
  icon bbox,
- the target shape id/name,
- the queried axis, extremum, selected row/column number, and winning count,
- row and column target-shape counts,
- selected-line target cells and off-line target distractor cells,
- render-style metadata for the panel title and row/column number-label text
  legibility,
- query, answer-line, grid-size, winning-count, shape, and fill-style
  probability metadata.
