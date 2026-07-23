# `task_games__slot_machine__winning_payline_count`

## Contract
1. Domain: `games`
2. Scene: `slot_machine`
3. Scene id: `slot_machine`
4. Public task id: `task_games__slot_machine__winning_payline_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer`
7. Annotation schema: `segment_set`

## Program Contract

Program: `count(payline for payline in rows_plus_long_diagonals if all_symbols_match(payline)); scene=slot_machine; scope=winning_payline_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `winning_payline_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `payline`, `rows_plus_long_diagonals`, `if`, `all_symbols_match`, `slot_machine`, `winning_payline_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `matching`

## Generation Notes
1. The scene renders a front-view toy slot machine with a 3x3 visible reel window.
2. Paylines are the three full rows plus the two long diagonals; columns are not paylines.
3. A payline wins only when all three visible symbols on that row or diagonal match.
4. The answer is balanced across `0..5` winning paylines by construction.
5. Annotation is an unordered segment set, one centerline segment for each winning payline.
6. Scalar annotation checked: true.
