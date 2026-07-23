# `task_games__battleship__last_ship_cell_label`

## Contract
1. Domain: `games`
2. Scene id: `battleship`
3. Public task id: `task_games__battleship__last_ship_cell_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string`
6. Annotation schema: `point`
7. Program schema: `label(select(candidate_cells, completes_only_not_sunk_ship)); scene=battleship; scope=last_ship_cell_label`

## Program Contract

Program: `label(select(candidate_cells, completes_only_not_sunk_ship)); scene=battleship; scope=last_ship_cell_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `last_ship_cell_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select`, `candidate_cells`, `completes_only_not_sunk_ship`, `battleship`, `last_ship_cell_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; the value is the unique visible candidate-cell label.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Generation Notes
1. The Battleship scene uses five fleet shapes: `Line 5`, `Line 4`, `Line 3`, `Square 2x2`, and `L 3`.
2. This task renders a hidden-ship tracking grid: red hit markers, gray miss markers, fleet-shape panel, and six labeled candidate cells `A-F`.
3. All non-target ships are fully hit. The target ship has exactly one unhit cell, and exactly one candidate label marks that cell.
4. Annotation is one point at the selected answer cell center.
