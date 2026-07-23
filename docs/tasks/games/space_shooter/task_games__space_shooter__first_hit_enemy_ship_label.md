# `task_games__space_shooter__first_hit_enemy_ship_label`

## Contract
1. Domain: `games`
2. Scene: `space_shooter`
3. Scene id: `space_shooter`
4. Public task id: `task_games__space_shooter__first_hit_enemy_ship_label`
5. Supported `query_id` values: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `bbox`

## Program Contract

Program: `select(label in visible_candidate_enemy_ships where distance_from_current_same_lane_player_shot_below is minimal); scene=space_shooter; scope=first_hit_enemy_ship_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `first_hit_enemy_ship_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `visible_candidate_enemy_ships`, `where`, `distance_from_current_same_lane_player_shot_below`, `is`, `minimal`, `space_shooter`, `first_hit_enemy_ship_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `state_update`

## Generation Notes
1. `single` is the only public query id; the task-specific prompt key is trace metadata.
2. Exactly four enemy ships are visibly labeled `A` through `D`.
3. Blue player shots move upward in their lane and hit the lower same-lane enemy ship first.
4. The four labeled ships have unique hit distances from the current blue shots, so exactly one labeled ship is hit first.
5. Annotation is the scalar bbox of the selected labeled enemy ship.
6. Unlabeled enemy ships and red enemy shots are visual distractors; only the labeled enemy ships are answer options.
7. Scalar annotation checked: true.
