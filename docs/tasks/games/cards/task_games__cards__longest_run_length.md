# `task_games__cards__longest_run_length`

## Contract
1. Domain: `games`
2. Scene id: `cards`
3. Public task id: `task_games__cards__longest_run_length`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `bbox_set`
7. Program schema: `max(lengths(consecutive_rank_runs(cards_in_display_order))); scene=cards; scope=longest_run_length`

## Program Contract

Program: `max(lengths(consecutive_rank_runs(cards_in_display_order))); scene=cards; scope=longest_run_length`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `longest_run_length` objective scope.
Operands: visible scene state and prompt-bound operands named by `lengths`, `consecutive_rank_runs`, `cards_in_display_order`, `cards`, `longest_run_length`.
Operation: evaluate `max` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## Generation Notes
2. Prompt wording comes from `src/trace_tasks/resources/prompts/games/cards/games_cards_v1.json`.
3. Annotation is projected from the same generated game state used for answer verification.
