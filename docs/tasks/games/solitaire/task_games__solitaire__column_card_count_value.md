# `task_games__solitaire__column_card_count_value`

## Contract
1. Domain: `games`
2. Scene id: `solitaire`
3. Public task id: `task_games__solitaire__column_card_count_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `point_set`
7. Program schema: `count(visible_cards_in_requested_tableau_column); scene=solitaire; scope=column_card_count_value`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count(visible_cards_in_requested_tableau_column); scene=solitaire; scope=column_card_count_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `column_card_count_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `visible_cards_in_requested_tableau_column`, `solitaire`, `column_card_count_value`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`

## Generation Notes
1. The prompt names a visible tableau column by its 1-based column number.
2. The answer is the number of visible cards in that column.
3. Annotation contains one point on the visible part of each card in the requested column.
4. Prompt wording comes from `src/trace_tasks/resources/prompts/games/solitaire/games_solitaire_v1.json`.
