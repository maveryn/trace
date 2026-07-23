# `task_games__slot_machine__paytable_score_value`

## Contract
1. Domain: `games`
2. Scene: `slot_machine`
3. Scene id: `slot_machine`
4. Public task id: `task_games__slot_machine__paytable_score_value`
5. Supported `query_id` values: `single`
6. Answer schema: `integer`
7. Annotation schema: `segment`

## Program Contract

Program: `paytable[matching_symbol(payline)] for the single payline in rows_plus_long_diagonals where all_symbols_match(payline); scene=slot_machine; scope=paytable_score_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `paytable_score_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `paytable`, `matching_symbol`, `payline`, `rows_plus_long_diagonals`, `if`, `all_symbols_match`, `slot_machine`, `paytable_score_value`.
Operation: evaluate `sum` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`, `matching`

## Generation Notes
1. The scene renders a front-view toy slot machine with a 3x3 visible reel window and a side paytable.
2. Paylines are the three full rows plus the two long diagonals; columns are not paylines.
3. A payline scores only when all three visible symbols on that row or diagonal match.
4. Generation samples exactly one scoring payline for this task.
5. The score for the winning payline is the side-paytable value for the matching symbol.
6. The answer is the integer score read from the side paytable.
7. Annotation is the scalar segment for the centerline of the scoring payline.
8. Scalar annotation checked: true.
