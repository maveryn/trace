# `task_games__hex__winning_move_cell_label`

## Contract
1. Domain: `games`
2. Scene: `hex`
3. Scene id: `hex`
4. Public task id: `task_games__hex__winning_move_cell_label`
5. Supported `query_id` values: `single`
6. Answer schema: `string_label`
7. Annotation schema: `point`
8. Program schema: `label(filter(labeled_empty_cells, move_result=connects_player_sides)); scene=hex; scope=winning_move_cell_label`

## Program Contract

Program: `label(filter(labeled_empty_cells, move_result=connects_player_sides)); scene=hex; scope=winning_move_cell_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `winning_move_cell_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `labeled_empty_cells`, `move_result`, `connects_player_sides`, `hex`, `winning_move_cell_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `topology`, `state_update`

## Generation Notes
1. `query_id=single` is the public no-branch query id; the prompt uses the Hex winning-move template.
2. Candidate labels are rendered in the image and the answer is the one label that wins immediately.
3. Annotation is the scalar point at the center of the winning candidate cell.
