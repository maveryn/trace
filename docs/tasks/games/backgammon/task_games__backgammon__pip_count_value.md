# `task_games__backgammon__pip_count_value`

## Contract
1. Domain: `games`
2. Scene id: `backgammon`
3. Public task id: `task_games__backgammon__pip_count_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `bbox_set`
7. Program schema: `sum(map(active_player_occupied_points, checker_count * distance_to_bear_off)); scene=backgammon; scope=pip_count_value`

## Program Contract

Program: `sum(map(active_player_occupied_points, checker_count * distance_to_bear_off)); scene=backgammon; scope=pip_count_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `pip_count_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `map`, `active_player_occupied_points`, `checker_count`, `distance_to_bear_off`, `backgammon`, `pip_count_value`.
Operation: evaluate `sum` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `topology`, `formula_evaluation`

## Generation Notes
1. Query ids are internal replay/sampling keys and do not define public task units.
2. The generated board is a sparse exact-answer race position with answer support `1..8`.
3. Pip distance follows standard Backgammon bearing-off distance: black checkers on point `p` contribute `p` each, while white checkers on point `p` contribute `25 - p` each.
4. Pip-count renders omit dice and add `D` labels beside active-player stacks to show the distance-to-bear-off value used in the sum.
5. Annotation is projected from active-player occupied point bboxes that contribute to the pip-count sum, not individual checker bboxes.
