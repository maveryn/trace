# `task_games__circular_chess__target_cell_reacher_count`

## Contract
1. Domain: `games`
2. Scene id: `circular_chess`
3. Public task id: `task_games__circular_chess__target_cell_reacher_count`
4. Supported `query_id` values: `white_piece_reaches_target_count`, `black_piece_reaches_target_count`
5. Answer schema: `integer_count`
6. Annotation schema: `point_set`
7. Program schema: `count(filter(pieces(target_color), target_cell in legal_destinations(piece))); scene=circular_chess; scope=target_cell_reacher_count`

## Program Contract

Program: `count(filter(pieces(target_color), target_cell in legal_destinations(piece))); scene=circular_chess; scope=target_cell_reacher_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `target_cell_reacher_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `pieces`, `target_color`, `target_cell`, `legal_destinations`, `piece`, `circular_chess`, `target_cell_reacher_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `white_piece_reaches_target_count`, `black_piece_reaches_target_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`

## Generation Notes
1. The blue marker identifies the target cell.
2. Pawns, check, checkmate, castling, en passant, and promotion are intentionally out of scope.
3. Annotation marks source-piece centers as pixel-space points.
