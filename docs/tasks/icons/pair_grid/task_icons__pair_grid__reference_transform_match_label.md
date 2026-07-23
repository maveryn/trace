# `task_icons__pair_grid__reference_transform_match_label`

## Identity
- domain: `icons`
- scene_id: `pair_grid`
- module: `src/trace_tasks/tasks/icons/pair_grid/reference_transform_match_label.py`
- prompt bundle: `icons_pair_grid_v1`

## Program Contract

Program: `selection.reference_relation_match(scene=pair_grid, scope=labeled_scene_cells, reference_relation=geometric_transform, output=option_letter)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `labeled_scene_cells` objective scope.
Operands: visible scene state and prompt-bound operands named by `pair_grid`, `labeled_scene_cells`, `reference_relation`, `geometric_transform`.
Operation: evaluate `selection.reference_relation_match` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Contract
Renders a Reference before/after icon pair and a labeled Scene grid of four
before/after icon-pair cells, then asks which labeled Scene cell shows the same
geometric transformation as the Reference pair. Exactly one Scene cell is
correct by construction.

Supported query ids:
- `single`

Answer schema: `option_letter`.
Annotation schema: scalar `bbox` around the selected Scene cell.

## Notes
The geometric transform id is sampled from
`rot90|rot180|rot270|flip_h|flip_v` and recorded as trace metadata, not as a
public query id. The task uses asymmetric icons so distractor transforms remain
visually distinct from the Reference transform.

Renderer metadata records sampled palette/style, panel-header and cell-label
text-legibility metadata, and per-icon noise edits.
