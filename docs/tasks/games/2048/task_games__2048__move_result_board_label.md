# `task_games__2048__move_result_board_label`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/2048/`
3. Scene id: `2048`
4. Public task id: `task_games__2048__move_result_board_label`
5. Supported `query_id` values: `single`
6. Prompt query key: `move_result_board_label`
7. Answer schema: `string_label`
8. Annotation schema: `bbox`
9. Program schema: `label(select_option(candidate_result_boards, option_board = simulate(board, rules=slide_merge_2048, action=move_direction).final_board)); scene=2048; scope=move_result_board_label`

## Program Contract

Program: `label(select_option(candidate_result_boards, option_board = simulate(board, rules=slide_merge_2048, action=move_direction).final_board)); scene=2048; scope=move_result_board_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `move_result_board_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select_option`, `candidate_result_boards`, `option_board`, `simulate`, `board`, `rules`, `slide_merge_2048`, `action`, `move_direction`, `final_board`, `move_result_board_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is the scalar bounding box around the single selected candidate board.
4. Annotation is projected from the same generated game state used for answer verification.
