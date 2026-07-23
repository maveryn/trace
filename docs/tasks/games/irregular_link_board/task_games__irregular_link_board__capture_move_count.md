# `task_games__irregular_link_board__capture_move_count`

## Program Contract

Program: `count(filter(empty_points, predicate=legal_jump_capture_destination(marked_piece, occupied_points, drawn_links))); scene=irregular_link_board; scope=capture_move_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `capture_move_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `empty_points`, `predicate`, `legal_jump_capture_destination`, `marked_piece`, `occupied_points`, `drawn_links`, `irregular_link_board`, `capture_move_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`, `state_update`

## Contract
- Domain: `games`
- Scene id: `irregular_link_board`
- Public task id: `task_games__irregular_link_board__capture_move_count`
- Supported `query_id` values: `single`
- Answer schema: `integer`
- Annotation schema: `point_set`

## Contract Notes
1. The board uses the same Fanorona/Alquerque-style base graph as the destination-count task: orthogonal links plus alternating diagonals, so diagonal lines do not cross unless the crossing is a playable point.
2. Public `query_id` is always `single`; the prompt template key remains `capture_move_count`.
3. A legal capture move jumps over one occupied adjacent point along a straight drawn line and lands on the empty point immediately beyond it.
4. Both links in that straight line must be drawn, and the landing point must be empty.
5. The capture task uses `5x5` and `6x6` boards with answer support `0..6`.
6. Annotation marks the centers of every legal capture destination point, and an empty annotation list is valid when the answer is `0`.
