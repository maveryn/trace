# `task_games__cards__poker_best_hand_label`

## Contract
1. Domain: `games`
2. Scene id: `cards`
3. Public task id: `task_games__cards__poker_best_hand_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string_label`
6. Annotation schema: `bbox_set`
7. Program schema: `label(arg_extreme(hands, metric=poker_hand_rank(hand), direction=best)); scene=cards; scope=poker_best_hand_label`

## Program Contract

Program: `label(arg_extreme(hands, metric=poker_hand_rank(hand), direction=best)); scene=cards; scope=poker_best_hand_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `poker_best_hand_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `arg_extreme`, `hands`, `metric`, `poker_hand_rank`, `hand`, `direction`, `best`, `cards`, `poker_best_hand_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`

## Generation Notes
2. Prompt wording comes from `src/trace_tasks/resources/prompts/games/cards/games_cards_v1.json`.
3. Annotation is projected from the same generated game state used for answer verification.
