# `task_games__battleship__remaining_ship_shape_label`

## Contract
1. Domain: `games`
2. Scene id: `battleship`
3. Public task id: `task_games__battleship__remaining_ship_shape_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string`
6. Annotation schema: `bbox`
7. Program schema: `label(select(shape_options, shape_id=untouched_ship_shape)); scene=battleship; scope=remaining_ship_shape_label`

## Program Contract

Program: `label(select(shape_options, shape_id=untouched_ship_shape)); scene=battleship; scope=remaining_ship_shape_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `remaining_ship_shape_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select`, `shape_options`, `shape_id`, `untouched_ship_shape`, `battleship`, `remaining_ship_shape_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; the value is the unique visible fleet-shape option label.
Annotation witnesses: `annotation` uses the `bbox` schema; one pixel-space bounding box marks the selected answer choice.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Generation Notes
1. The Battleship scene uses five fleet shapes: `Line 5`, `Line 4`, `Line 3`, `Square 2x2`, and `L 3`.
2. This task renders a hidden-ship tracking grid: red hit markers, gray miss markers, and five labeled answer choices.
3. Exactly four ships are fully hit. The remaining target ship is untouched, with no red hit markers on any of its cells.
4. The option panel contains all five fleet-shape choices, exactly one of which matches the untouched ship shape.
5. Annotation is one pixel-space bounding box around the selected answer choice.
