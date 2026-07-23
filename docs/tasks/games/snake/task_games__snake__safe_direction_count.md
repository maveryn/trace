# `task_games__snake__safe_direction_count`

## Contract
1. Domain: `games`
2. Scene id: `snake`
3. Public task id: `task_games__snake__safe_direction_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(filter(cardinal_directions, snake_next_cell_is_safe)); scene=snake; scope=safe_direction_count`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count(filter(cardinal_directions, snake_next_cell_is_safe)); scene=snake; scope=safe_direction_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `safe_direction_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `cardinal_directions`, `snake_next_cell_is_safe`, `snake`, `safe_direction_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. Count the immediate up/down/left/right moves that keep the head inside the board and out of the body or gray walls.
2. Annotation is the bbox set for safe destination cells. It is empty when no direction is safe.
3. Prompt wording comes from `src/trace_tasks/resources/prompts/games/snake/games_snake_v1.json`.
