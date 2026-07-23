# `task_games__snakes_ladders__special_square_count`

## Contract
1. Domain: `games`
2. Scene id: `snakes_ladders`
3. Public task id: `task_games__snakes_ladders__special_square_count`
4. Supported `query_id` values: `ladder_count`, `snake_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(visible_jump_starts(kind=query_kind)); scene=snakes_ladders; scope=special_square_count`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count(visible_jump_starts(kind=query_kind)); scene=snakes_ladders; scope=special_square_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `special_square_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `visible_jump_starts`, `kind`, `query_kind`, `snakes_ladders`, `special_square_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `ladder_count`, `snake_count`.

## Reasoning Operations

Families: `counting`

## Generation Notes
1. Count every visible ladder start or every visible snake head, depending on the query.
2. Annotation is the bbox set for all counted ladder-start or snake-head squares.
3. Prompt wording comes from `src/trace_tasks/resources/prompts/games/snakes_ladders/games_snakes_ladders_v1.json`.
