# `task_games__sixteen_soldiers__marked_piece_capture_count`

## Program Contract

Program: `count(immediate_jump_captures(x_marked_piece, drawn_line_graph)); scene=sixteen_soldiers; scope=marked_piece_capture_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_piece_capture_count` objective scope.
Operands: visible scene state and prompt-bound target operands named by the task contract.
Operation: evaluate `count(immediate_jump_captures(x_marked_piece, drawn_line_graph))` over the visible Sixteen Soldiers board graph; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema for capturable opponent piece centers.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`, `state_update`

## Generation Notes

- The board is a fixed 37-point Sixteen Soldiers line graph with a 5 by 5 center and two triangular extensions.
- A capture is counted only when the X-marked piece can jump in one drawn straight line over an adjacent opponent piece and land on the empty point immediately beyond.
- Annotation marks the centers of capturable opponent pieces. Landing points remain in trace metadata for verifier/debugging, but are not prompt-facing annotation.
- The answer and annotation are bound from the same generated board state.
