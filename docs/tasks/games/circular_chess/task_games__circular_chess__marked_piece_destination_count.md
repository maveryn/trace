# `task_games__circular_chess__marked_piece_destination_count`

## Contract
1. Domain: `games`
2. Scene id: `circular_chess`
3. Public task id: `task_games__circular_chess__marked_piece_destination_count`
4. Supported `query_id` values: `marked_piece_move_count`, `marked_piece_capture_count`
5. Answer schema: `integer_count`
6. Annotation schema: `point_set`
7. Program schema: `count(filter(legal_destinations(marked_piece), destination_filter)); scene=circular_chess; scope=marked_piece_destination_count`

## Program Contract

Program: `count(filter(legal_destinations(marked_piece), destination_filter)); scene=circular_chess; scope=marked_piece_destination_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_piece_destination_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `legal_destinations`, `marked_piece`, `destination_filter`, `circular_chess`, `marked_piece_destination_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `marked_piece_move_count`, `marked_piece_capture_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`

## Generation Notes
1. The scene uses a four-ring by sixteen-sector circular board with sector wraparound.
2. Pawns, check, checkmate, castling, en passant, and promotion are intentionally out of scope.
3. Annotation marks destination-cell centers as pixel-space points.
