# `task_games__pool__blocking_ball_count`

## Program Contract

Program: `count(filter(pool_balls, intersects_marked_shot_segment)); scene=pool; scope=blocking_ball_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `blocking_ball_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `pool_balls`, `intersects_marked_shot_segment`, `pool`, `blocking_ball_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; it contains the visible blocking balls that intersect the marked shot segments.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Generation Notes

- The shown shot has two straight segments: cue ball to the marked target ball, then the marked target ball to the marked pocket.
- Annotation is a `bbox_set` projected around the blocking balls from the same generated pool state used for answer verification.
