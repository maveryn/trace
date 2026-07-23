# `task_games__tetris__line_clear_count`

## Contract
1. Domain: `games`
2. Scene: `tetris`
3. Scene id: `tetris`
4. Public task id: `task_games__tetris__line_clear_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_map`

## Program Contract

Program: `max(cleared_row_count(simulate_drop(board, rotate_translate(next_piece, placement)))) over legal placements; scene=tetris; scope=line_clear_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `line_clear_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `cleared_row_count`, `simulate_drop`, `board`, `rotate_translate`, `next_piece`, `placement`, `tetris`, `line_clear_count`.
Operation: evaluate `max` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `state_update`

## Generation Notes
1. `single` is the only public query id; the task-specific prompt key is trace metadata.
2. Annotation maps `board` and `next_piece` to the rendered board and NEXT-piece preview boxes.
3. Scalar annotation checked: true.
