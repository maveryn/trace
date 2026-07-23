# `task_icons__named_grid__line_adjacency_pair_count`

- domain: `icons`
- scene_id: `named_grid`
- task: `line_adjacency_pair_count`
- module: `src/trace_tasks/tasks/icons/named_grid/line_adjacency_pair_count.py`

## Program Contract

Program: `count.adjacent_pair(scene=named_grid, scope=numbered_row_or_column, adjacency=orthogonal_consecutive, pair_order=unordered, attributes=shape_pair, output=count)`

Candidate set: adjacent two-cell windows along the prompt-addressed numbered row or column.
Operands: visible scene state and prompt-bound operands named by `named_grid`, `numbered_row_or_column`, `orthogonal_consecutive`, `unordered`, and the two shape labels in `shape_pair`.
Operation: count every adjacent two-cell window in the selected row or column whose two shape ids match the requested unordered pair. Direction does not matter, so `A-B` and `B-A` both match. Overlapping windows count separately, so `A-B-A` contributes two matching adjacent pairs.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; each segment connects the centers of one counted adjacent pair.
Annotation schema: `segment_set`.
Query ids: `row_unordered_adjacent_pair_count`, `column_unordered_adjacent_pair_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Query IDs

- `row_unordered_adjacent_pair_count`
- `column_unordered_adjacent_pair_count`

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

The selected grid line is constructed to contain exactly the sampled number of
matching adjacent pair windows. Non-matching positions in that line use other
icon types to prevent accidental extra matches. Other rows or columns may
contain the same icon types as distractors because only the prompt-addressed
line is in scope.

Fill style and color are rendered as non-semantic visual variation.

## Trace

The trace records:

- every named icon with row/column indices, row/column numbers, cell bbox, and
  icon bbox,
- the two target shape ids/names,
- the queried axis and one-based row/column number,
- the counted adjacent cell pairs and their projected center-to-center segment
  annotations,
- render-style metadata for row/column number-label text legibility,
- query, answer, grid-size, line-index, shape, and fill-style probability
  metadata.
