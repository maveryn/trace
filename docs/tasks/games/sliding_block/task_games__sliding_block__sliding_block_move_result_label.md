# `task_games__sliding_block__sliding_block_move_result_label`

## Contract
1. Domain: `games`
2. Scene id: `sliding_block`
3. Public task id: `task_games__sliding_block__sliding_block_move_result_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox_map`
7. Program schema: `select(option_board_equal_to(apply_ordered_slides(source_board, slide_sequence))); scene=sliding_block; scope=sliding_block_move_result_label`
8. Scalar annotation checked: `true`

## Program Contract

Program: `select(option_board_equal_to(apply_ordered_slides(source_board, slide_sequence))); scene=sliding_block; scope=sliding_block_move_result_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `sliding_block_move_result_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `option_board_equal_to`, `apply_ordered_slides`, `source_board`, `slide_sequence`, `sliding_block`, `sliding_block_move_result_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`, `matching`

## Generation Notes
1. The prompt gives a short ordered slide sequence on a neutral board with no target block.
2. The image always shows exactly four visual option boards.
3. The answer is the visual option label whose board matches the final state.
4. Annotation is a role-keyed bbox map with `source_board` for the original board and `selected_option` for the correct option panel.
