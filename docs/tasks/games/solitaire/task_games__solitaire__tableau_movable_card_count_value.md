# `task_games__solitaire__tableau_movable_card_count_value`

## Contract
1. Domain: `games`
2. Scene id: `solitaire`
3. Public task id: `task_games__solitaire__tableau_movable_card_count_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(exposed_tableau_cards(with_legal_tableau_destination)); scene=solitaire; scope=tableau_movable_card_count_value`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count(exposed_tableau_cards(with_legal_tableau_destination)); scene=solitaire; scope=tableau_movable_card_count_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `tableau_movable_card_count_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `exposed_tableau_cards`, `with_legal_tableau_destination`, `solitaire`, `tableau_movable_card_count_value`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
1. The scene shows tableau columns only; foundation piles are hidden for this tableau-only objective.
2. Count exposed tableau cards that can legally move onto at least one other exposed tableau card.
3. A legal tableau move places the moving card on a target card that is exactly one rank higher and the opposite color.
4. Empty columns and foundation piles are not considered for this objective.
5. Annotation contains the bboxes for the counted exposed tableau cards.
6. Prompt wording comes from `src/trace_tasks/resources/prompts/games/solitaire/games_solitaire_v1.json`.
