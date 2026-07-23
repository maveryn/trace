# `task_games__chess__target_square_attacker_count`

## Contract
1. Domain: `games`
2. Scene id: `chess`
3. Public task id: `task_games__chess__target_square_attacker_count`
4. Supported `query_id` values: `king_square_attacker_count`, `white_piece_attacks_target_square_count`, `black_piece_attacks_target_square_count`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(attackers(marked_target_square, queried_side)); scene=chess; scope=target_square_attacker_count`

## Program Contract

Program: `count(attackers(marked_target_square, queried_side)); scene=chess; scope=target_square_attacker_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `target_square_attacker_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `attackers`, `marked_target_square`, `queried_side`, `chess`, `target_square_attacker_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `king_square_attacker_count`, `white_piece_attacks_target_square_count`, `black_piece_attacks_target_square_count`.

## Reasoning Operations

Families: `counting`, `spatial_relations`

## Generation Notes
1. Annotation marks bounding boxes for all attacking pieces from the queried side.
2. For `king_square_attacker_count`, the marked target square contains the king and the queried side is the opponent.
3. For the white/black target-square queries, the marked target square is empty.
