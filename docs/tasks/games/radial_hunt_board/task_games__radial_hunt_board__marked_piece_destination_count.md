# `task_games__radial_hunt_board__marked_piece_destination_count`

## Contract
1. Domain: `games`
2. Scene: `radial_hunt_board`
3. Scene id: `radial_hunt_board`
4. Public task id: `task_games__radial_hunt_board__marked_piece_destination_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `point_set`
8. Program schema: `count(simple_adjacent_empty_points(linked_to(x_marked_piece))); scene=radial_hunt_board; scope=marked_piece_destination_count`

## Program Contract

Program: `count(simple_adjacent_empty_points(linked_to(x_marked_piece))); scene=radial_hunt_board; scope=marked_piece_destination_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_piece_destination_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `simple_adjacent_empty_points`, `linked_to`, `x_marked_piece`, `radial_hunt_board`, `marked_piece_destination_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`, `state_update`

## Generation Notes
1. The board is a Pretwa-inspired radial graph with three concentric circles and three diameters, producing 19 playable points.
2. A simple-move destination is an adjacent empty point connected to the X-marked piece along one drawn circle or diameter segment.
3. Occupied adjacent points and capture-jump landing points are excluded.
4. Capture jumps are covered by `task_games__radial_hunt_board__capture_move_count`.
5. The answer range is `0..6`; annotation marks the centers of every simple-move destination point, and an empty annotation list is valid when the answer is `0`.
