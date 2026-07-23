# `task_games__sixteen_soldiers__marked_piece_destination_count`

## Program Contract

Program: `count(simple_adjacent_empty_destinations(x_marked_piece, drawn_line_graph)); scene=sixteen_soldiers; scope=marked_piece_destination_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_piece_destination_count` objective scope.
Operands: visible scene state and prompt-bound target operands named by the task contract.
Operation: evaluate `count(simple_adjacent_empty_destinations(x_marked_piece, drawn_line_graph))` over the visible Sixteen Soldiers board graph; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema for simple adjacent empty destination centers.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`, `state_update`

## Generation Notes

- The board is a fixed 37-point Sixteen Soldiers line graph with a 5 by 5 center and two triangular extensions.
- A simple-move destination is an adjacent empty point connected to the X-marked piece by one drawn line.
- Jump and capture landing points are excluded; captures are covered by `task_games__sixteen_soldiers__marked_piece_capture_count`.
- The generated answer support is `0..5`, with `6..10` pieces per side.
- Annotation marks the centers of every simple-move destination point; an empty annotation list is valid when the answer is `0`.
- The answer and annotation are bound from the same generated board state.
