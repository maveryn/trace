# `task_games__space_shooter__hit_enemy_ship_label`

## Contract
1. Domain: `games`
2. Scene: `space_shooter`
3. Scene id: `space_shooter`
4. Public task id: `task_games__space_shooter__hit_enemy_ship_label`
5. Supported `query_id` values: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `bbox`

## Program Contract

Program: `select(label in visible_candidate_enemy_ships where enemy_ship_id in lower_first_hits_by_lane(blue_player_shots, enemy_ships)); scene=space_shooter; scope=hit_enemy_ship_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `hit_enemy_ship_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `visible_candidate_enemy_ships`, `where`, `enemy_ship_id`, `lower_first_hits_by_lane`, `blue_player_shots`, `enemy_ships`, `space_shooter`, `hit_enemy_ship_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `state_update`

## Generation Notes
1. `single` is the only public query id; the task-specific prompt key is trace metadata.
2. Exactly four candidate enemy ships are visibly labeled `A` through `D`.
3. Blue player shots move upward in their lane; each visible blue shot can destroy one enemy ship above it in that same lane.
4. If a lane has fewer blue shots than enemy ships, the lower enemy ships are destroyed first.
5. Exactly one labeled candidate ship is hit under that rule; unlabeled ships may also be hit.
6. Annotation is the scalar bbox of the selected labeled enemy ship.
7. Scalar annotation checked: true.
