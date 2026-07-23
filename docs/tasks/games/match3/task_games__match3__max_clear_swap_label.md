# `task_games__match3__max_clear_swap_label`

## Contract
1. Domain: `games`
2. Scene id: `match3`
3. Public task id: `task_games__match3__max_clear_swap_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `point`

## Program Contract

Program: `selection.extreme_metric_label(candidate_set=visible_swap_arrows, metric=immediate_clear_count_after_arrow_swap, direction=maximum); scene=match3; scope=max_clear_swap_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `max_clear_swap_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `candidate_set`, `visible_swap_arrows`, `metric`, `immediate_clear_count_after_arrow_swap`, `direction`, `maximum`, `match3`, `max_clear_swap_label`.
Operation: evaluate `selection.extreme_metric_label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `state_update`

## Generation Notes
1. The selected arrow is the only displayed option with the largest immediate clear count.
2. The immediate clear rule counts horizontal or vertical runs of three or more after the swap; no falling, refill, special effects, or cascades are applied.
3. Exactly four labeled swap options are shown.
4. Annotation is the scalar point on the selected swap arrow.
