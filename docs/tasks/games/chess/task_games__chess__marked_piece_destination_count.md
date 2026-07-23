# `task_games__chess__marked_piece_destination_count`

## Contract
1. Domain: `games`
2. Scene id: `chess`
3. Public task id: `task_games__chess__marked_piece_destination_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(empty_legal_destinations(marked_piece)); scene=chess; scope=marked_piece_destination_count`

## Program Contract

Program: `count(empty_legal_destinations(marked_piece)); scene=chess; scope=marked_piece_destination_count`

Candidate set: empty board squares reachable by the marked piece in one normal chess move.
Operands: the marked chess piece, visible board occupancy, friendly blockers, and normal chess movement rules.
Operation: count legal one-move destination squares that are empty; capturable opponent-occupied squares are excluded.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses `bbox_set`, one board-square box for each counted empty destination square.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
1. The red outlined square contains the marked piece.
2. Friendly pieces block movement.
3. Opponent pieces are capture targets and are not counted by this task.
4. Capturable opponent pieces are covered by `task_games__chess__marked_piece_capture_count`.
5. The internal prompt/query key is `marked_piece_destination_count`; it is not a public query branch.
