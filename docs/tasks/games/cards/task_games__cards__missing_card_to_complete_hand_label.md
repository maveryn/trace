# `task_games__cards__missing_card_to_complete_hand_label`

## Contract
1. Domain: `games`
2. Scene id: `cards`
3. Public task id: `task_games__cards__missing_card_to_complete_hand_label`
4. Supported `query_id` values: `missing_flush_card_label`, `missing_straight_card_label`, `missing_full_house_card_label`, `missing_three_of_kind_card_label`
5. Answer schema: `string_label`
6. Annotation schema: `bbox`
7. Program schema: `label(select(candidate_cards, completes_pattern(partial_hand, candidate_card, target_pattern))); scene=cards; scope=missing_card_to_complete_hand_label`

## Program Contract

Program: `label(select(candidate_cards, completes_pattern(partial_hand, candidate_card, target_pattern))); scene=cards; scope=missing_card_to_complete_hand_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `missing_card_to_complete_hand_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select`, `candidate_cards`, `completes_pattern`, `partial_hand`, `candidate_card`, `target_pattern`, `cards`, `missing_card_to_complete_hand_label` plus the active `query_id` branch.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `missing_flush_card_label`, `missing_straight_card_label`, `missing_full_house_card_label`, `missing_three_of_kind_card_label`.

## Reasoning Operations

Families: `matching`

## Generation Notes
2. Query ids are internal pattern branches inside the same public task contract.
3. Prompt wording comes from `src/trace_tasks/resources/prompts/games/cards/games_cards_v1.json`.
4. Annotation is the selected candidate-card bbox projected from the same generated card state used for answer verification.
