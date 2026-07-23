# `task_games__irregular_link_board__marked_piece_destination_count`

## Program Contract

Program: `count(filter(empty_points, predicate=adjacent_linked_destination(marked_piece, drawn_links))); scene=irregular_link_board; scope=marked_piece_destination_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_piece_destination_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `empty_points`, `predicate`, `adjacent_linked_destination`, `marked_piece`, `drawn_links`, `irregular_link_board`, `marked_piece_destination_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`, `state_update`

## Contract
- Domain: `games`
- Scene id: `irregular_link_board`
- Public task id: `task_games__irregular_link_board__marked_piece_destination_count`
- Supported `query_id` values: `single`
- Answer schema: `integer`
- Annotation schema: `point_set`

## Contract Notes
1. Public `query_id` is always `single`; the prompt template key remains `marked_piece_destination_count`.
2. A legal destination is an adjacent empty point connected to the X-marked piece by exactly one drawn link.
3. Adjacent occupied points and adjacent points without a drawn link are not legal destinations.
4. The answer range is `0..6`.
5. Annotation marks the centers of every legal empty destination point, and an empty annotation list is valid when the answer is `0`.

## Scene Notes
1. The board is a variable-size point lattice with a Fanorona/Alquerque-style base graph: orthogonal links plus alternating diagonals, so diagonal lines do not cross unless the crossing is a playable point.
