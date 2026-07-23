# `task_games__darts__dart_score_value`

## Contract
1. Domain: `games`
2. Scene: `darts`
3. Scene id: `darts`
4. Public task id: `task_games__darts__dart_score_value`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_value`
7. Annotation schema: `point`
8. Program schema: `value(score(sole_visible_dart)); scene=darts; scope=dart_score_value`

## Program Contract

Program: `value(score(sole_visible_dart)); scene=darts; scope=dart_score_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `dart_score_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `score`, `sole_visible_dart`, `darts`, `dart_score_value`.
Operation: evaluate `value` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Generation Notes
1. The scene renders a simplified dartboard with 10 numbered sectors, one center bullseye, and exactly one dart.
2. A dart in a numbered sector scores that number; a dart in the center bullseye scores `50`.
3. Query ids are internal replay/sampling keys and do not define public task units.
4. Annotation is the center point of the only visible dart, projected from the same generated game state used for answer verification.
