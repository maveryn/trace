# `task_games__space_shooter__enemy_ship_hit_count`

## Contract
1. Domain: `games`
2. Scene: `space_shooter`
3. Scene id: `space_shooter`
4. Public task id: `task_games__space_shooter__enemy_ship_hit_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `sum_by_lane(min(count(blue_player_shots), count(enemy_ships))); scene=space_shooter; scope=enemy_ship_hit_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `enemy_ship_hit_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `min`, `blue_player_shots`, `enemy_ships`, `space_shooter`, `enemy_ship_hit_count`.
Operation: evaluate `sum_by_lane` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `aggregation`

## Generation Notes
1. `single` is the only public query id; the task-specific prompt key is trace metadata.
2. Blue player shots move upward in their lane; each visible blue shot can destroy one enemy ship above it in that same lane.
3. A lane can contain zero to three blue player shots and zero to three enemy ships.
4. If a lane has fewer blue shots than enemy ships, the lower enemy ships are destroyed first; this makes the annotated ship set unique.
5. The sampler annotates the destroyed enemy ship bounding boxes.
6. Red enemy shots and lanes without usable blue shots are visual distractors.
7. Scalar annotation checked: true.
