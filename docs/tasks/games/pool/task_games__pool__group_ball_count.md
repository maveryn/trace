# `task_games__pool__group_ball_count`

## Program Contract

Program: `count(filter(pool_balls, group=current_player_group)); scene=pool; scope=group_ball_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `group_ball_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `pool_balls`, `group`, `current_player_group`, `pool`, `group_ball_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; it contains all visible balls in the current player's group.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes

- The current player group is either `solids` or `stripes`; the sampled group is trace metadata, not a public query branch.
- Annotation is a `bbox_set` projected around the matching balls from the same generated pool state used for answer verification.
