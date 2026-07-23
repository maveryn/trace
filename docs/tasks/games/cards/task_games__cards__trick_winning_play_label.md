# `task_games__cards__trick_winning_play_label`

## Contract
1. Domain: `games`
2. Scene id: `cards`
3. Public task id: `task_games__cards__trick_winning_play_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string_label`
6. Annotation schema: `bbox`
7. Program schema: `label(select(candidate_cards, would_win_trick(played_cards, candidate_card, led_suit, trump_suit))); scene=cards; scope=trick_winning_play_label`

## Program Contract

Program: `label(select(candidate_cards, would_win_trick(played_cards, candidate_card, led_suit, trump_suit))); scene=cards; scope=trick_winning_play_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `trick_winning_play_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select`, `candidate_cards`, `would_win_trick`, `played_cards`, `candidate_card`, `led_suit`, `trump_suit`, `cards`, `trick_winning_play_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
2. Prompt wording comes from `src/trace_tasks/resources/prompts/games/cards/games_cards_v1.json`.
3. Annotation is the selected candidate-card bbox projected from the same generated card state used for answer verification.
