# `task_games__snakes_ladders__move_outcome_value`

## Contract
1. Domain: `games`
2. Scene id: `snakes_ladders`
3. Public task id: `task_games__snakes_ladders__move_outcome_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_map`
7. Program schema: `value(simulate(start_square, rules=snakes_ladders_jumps, action=shown_die).final_square); scene=snakes_ladders; scope=move_outcome_value`
8. Scalar annotation checked: `true`

## Program Contract

Program: `value(simulate(start_square, rules=snakes_ladders_jumps, action=shown_die).final_square); scene=snakes_ladders; scope=move_outcome_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `move_outcome_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `simulate`, `start_square`, `rules`, `snakes_ladders_jumps`, `action`, `shown_die`, `final_square`, `snakes_ladders`, `move_outcome_value`.
Operation: evaluate `value` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. Move the token by the shown die value, then immediately follow a snake or ladder if the landing square starts one.
2. Annotation is a bbox map with `start_square` and `end_square` roles.
3. Prompt wording comes from `src/trace_tasks/resources/prompts/games/snakes_ladders/games_snakes_ladders_v1.json`.
