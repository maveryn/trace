# `task_games__solitaire__foundation_ready_count`

## Contract
1. Domain: `games`
2. Scene id: `solitaire`
3. Public task id: `task_games__solitaire__foundation_ready_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(exposed_tableau_cards(can_move_to_foundation)); scene=solitaire; scope=foundation_ready_count`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count(exposed_tableau_cards(can_move_to_foundation)); scene=solitaire; scope=foundation_ready_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `foundation_ready_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `exposed_tableau_cards`, `can_move_to_foundation`, `solitaire`, `foundation_ready_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
1. Count only exposed tableau cards, using the visible foundation suit and top-rank state.
2. Annotation contains only the bboxes for the counted exposed cards.
3. Prompt wording comes from `src/trace_tasks/resources/prompts/games/solitaire/games_solitaire_v1.json`.
