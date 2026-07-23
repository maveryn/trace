# `task_games__snake__path_outcome_option_label`

## Contract
1. Domain: `games`
2. Scene id: `snake`
3. Public task id: `task_games__snake__path_outcome_option_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox_set`
7. Program schema: `selection.option_value_match(simulate_snake_move_sequence, visible_result_options); scene=snake; scope=path_outcome_option_label`
8. Scalar annotation checked: `true`

## Program Contract

Program: `selection.option_value_match(simulate_snake_move_sequence, visible_result_options); scene=snake; scope=path_outcome_option_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `path_outcome_option_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `simulate_snake_move_sequence`, `visible_result_options`, `snake`, `path_outcome_option_label`.
Operation: evaluate `selection.option_value_match` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `state_update`, `matching`

## Generation Notes
1. Simulate the listed moves in order. The answer is the visible option label for the final head cell or `GAME OVER`.
2. Annotation is the bbox set for visible in-board cells traversed by the head up to the result.
3. The image always shows options `A` through `D`, with one `GAME OVER` option card.
4. Prompt wording comes from `src/trace_tasks/resources/prompts/games/snake/games_snake_v1.json`.
