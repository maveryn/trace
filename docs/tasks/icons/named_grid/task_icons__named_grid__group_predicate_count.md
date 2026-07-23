# `task_icons__named_grid__group_predicate_count`

- domain: `icons`
- scene_id: `named_grid`
- task: `group_predicate_count`
- module: `src/trace_tasks/tasks/icons/named_grid/group_predicate_count.py`

## Program Contract

Program: `count.group_predicate(scene=named_grid, scope=numbered_rows_or_columns, groups=grid_lines, predicates=at_least|exactly|none, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `numbered_rows_or_columns` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_grid`, `numbered_rows_or_columns`, `groups`, `grid_lines`, `predicates`, `at_least`, `exactly`, `none`.
Operation: evaluate `count.group_predicate` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Annotation schema: `bbox_set`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query IDs
- `row_at_least_shape_count`
- `column_at_least_shape_count`
- `row_exactly_shape_count`
- `column_exactly_shape_count`
- `row_no_shape_count`
- `column_no_shape_count`

## Generation
Default answer support is `0..5`. Grid sizes are sampled from every row/column
combination in `4..6`.

For at-least queries, thresholds are sampled from `2..3`. For exactly queries,
thresholds are sampled from `1..3`. No-shape queries use threshold `0`.

The generator first samples the number of qualifying rows or columns, then
constructs target-shape counts so exactly that many lines satisfy the selected
condition. Fill style and color are rendered as non-semantic visual variation.

## Trace
The trace records:
- every named icon with row/column indices, row/column numbers, cell bbox, and
  icon bbox,
- the target shape id/name,
- the queried axis, condition, threshold, and answer count,
- row and column target-shape counts,
- qualifying row/column indices, numbers, and region bboxes,
- render-style metadata for the panel title and row/column number-label text
  legibility,
- query, answer, grid-size, threshold, shape, and fill-style probability
  metadata.
