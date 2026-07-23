# `task_games__chess_variant__marked_piece_capture_count`

## Contract
1. Domain: `games`
2. Scene id: `chess_variant`
3. Public task id: `task_games__chess_variant__marked_piece_capture_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(capturable_opponent_pieces(marked_piece)); scene=chess_variant; scope=marked_piece_capture_count`

## Program Contract

Program: `count(capturable_opponent_pieces(marked_piece)); scene=chess_variant; scope=marked_piece_capture_count`

Candidate set: opponent-occupied board squares that the red-outlined marked piece can legally reach under the displayed movement rule.
Operands: the marked piece, occupied squares, friendly blockers, opponent pieces, the displayed rule family, and the displayed range or jump rule.
Operation: count opponent pieces on legal capture destinations. Empty movement squares and friendly-occupied blocked squares are excluded.
Output binding: `answer` uses the `integer` schema; generation binds a unique final count.
Annotation witnesses: `annotation` uses the `bbox_set` schema with one square bbox for each capturable opponent piece.
Query ids: `single` public query; internal prompt key `marked_piece_capture_count`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
1. The visible rule card defines the movement geometry for every piece.
2. Capture count is sampled independently from scene style and clutter axes.
3. Annotation is projected from the same generated board state used for answer verification.
