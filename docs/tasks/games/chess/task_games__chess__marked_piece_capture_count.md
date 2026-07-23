# `task_games__chess__marked_piece_capture_count`

## Contract
1. Domain: `games`
2. Scene id: `chess`
3. Public task id: `task_games__chess__marked_piece_capture_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(capturable_opponent_pieces(marked_piece)); scene=chess; scope=marked_piece_capture_count`

## Program Contract

Program: `count(capturable_opponent_pieces(marked_piece)); scene=chess; scope=marked_piece_capture_count`

Candidate set: opponent-occupied board squares reachable by the marked piece in one normal chess move.
Operands: the marked chess piece, visible board occupancy, friendly blockers, opponent pieces, and normal chess movement rules.
Operation: count opponent pieces that the marked piece can capture in one move; empty movement destinations are excluded.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses `bbox_set`, one board-square box for each capturable opponent piece.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
1. The red outlined square contains the marked piece.
2. Friendly pieces block movement.
3. Empty movement destinations are covered by `task_games__chess__marked_piece_destination_count`.
4. The internal prompt/query key is `marked_piece_capture_count`; it is not a public query branch.
