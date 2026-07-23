# `task_games__tetris__drop_result_label`

## Contract
1. Domain: `games`
2. Scene: `tetris`
3. Scene id: `tetris`
4. Public task id: `task_games__tetris__drop_result_label`
5. Supported `query_id` values: `single`
6. Answer schema: `string_label`
7. Annotation schema: `bbox`

## Program Contract

Program: `label(option where option.board = simulate_fixed_drop(board, falling_piece, line_clear_rules)); scene=tetris; scope=drop_result_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `drop_result_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `where`, `board`, `simulate_fixed_drop`, `falling_piece`, `line_clear_rules`, `tetris`, `drop_result_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. The target clear-count branch is a generation axis recorded as `target_clear_count`, not a public query branch.
2. The renderer always shows exactly four labeled result-board options in a two-by-two grid below the START board.
3. Annotation is the scalar bbox of the selected result-board option panel.
4. Scalar annotation checked: true.
