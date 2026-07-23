# task_games__ludo_board__winning_roll_value

Public taxonomy: `games -> ludo_board -> task_games__ludo_board__winning_roll_value`.

## Program Contract

Program: `exact_finish_roll(token_position, finish_cell); scene=ludo_board; scope=winning_roll_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `winning_roll_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `token_position`, `finish_cell`, `ludo_board`, `winning_roll_value`.
Operation: evaluate `exact_finish_roll` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation schema: `point`.
Annotation witnesses: `annotation` marks the queried token center.
Query ids: `single`.

The finish cell is the final colored home-lane cell near the center for the queried color. The answer is the exact single die roll needed to move from the queried token's current home-lane cell to that finish cell.

## Reasoning Operations

Families: `topology`, `state_update`, `formula_evaluation`

## Generator

- Implementation: `src/trace_tasks/tasks/games/ludo_board/winning_roll_value.py`
- Config: `src/trace_tasks/resources/configs/domains/games/ludo_board.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/games/ludo_board/games_ludo_board_v1.json`
