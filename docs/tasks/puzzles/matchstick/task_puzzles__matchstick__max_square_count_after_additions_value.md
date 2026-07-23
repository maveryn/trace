# `task_puzzles__matchstick__max_square_count_after_additions_value`

## Program Contract

Program: `count_max(matchstick_lattice.complete_unit_squares_after_k_edge_additions, add_count=k, source=visible_incomplete_square_lattice); scene=matchstick; scope=square_lattice_missing_edges`

Candidate set: the visible matchstick segments, digit/equation/lattice structure, segment labels, and labeled candidate options when present inside the `square_lattice_missing_edges` objective scope.
Operands: visible scene state and prompt-bound operands named by `matchstick_lattice`, `complete_unit_squares_after_k_edge_additions`, `add_count`, `k`, `source`, `visible_incomplete_square_lattice`, `matchstick`, `square_lattice_missing_edges`.
Operation: evaluate `count_max` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `state_update`

## Answer And Annotation

- Answer type: `integer`.
- Annotation type: `bbox_set`.
- Annotation schema: `bbox_set`.
- Annotation value: an unordered list of image-pixel unit-square bounding boxes `[x0, y0, x1, y1]`, one for each complete unit square counted in the final optimal board.
- The answer and annotation are bound from the same sampled trace. The sampler rejects instances unless the maximum answer and final counted square-bbox set are unique.

## Rendering And Prompt

The `matchstick` scene renders one incomplete matchstick lattice on a square grid using wooden-match, colored-rod, chalk-stick, neon-rod, or metal-rod visual styles. Existing sticks are rendered as matchsticks, and empty grid edge positions are indicated by the lattice vertices. Prompt prose comes from `src/trace_tasks/resources/prompts/puzzles/matchstick/puzzles_matchstick_v1.json`.
