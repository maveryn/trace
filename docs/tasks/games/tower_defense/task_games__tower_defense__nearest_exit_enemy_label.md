# `task_games__tower_defense__nearest_exit_enemy_label`

## Contract
1. Domain: `games`
2. Scene: `tower_defense`
3. Scene id: `tower_defense`
4. Public task id: `task_games__tower_defense__nearest_exit_enemy_label`
5. Supported `query_id` values: `single`
6. Answer schema: `label`
7. Annotation schema: `point`

## Program Contract

Program: `argmax_label(enemy in labeled_enemies_A_to_F, path_index(enemy)); scene=tower_defense; scope=nearest_exit_enemy_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `nearest_exit_enemy_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `enemy`, `labeled_enemies_A_to_F`, `path_index`, `tower_defense`, `nearest_exit_enemy_label`.
Operation: evaluate `argmax_label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `topology`

## Generation Notes
1. The path has six labeled enemy markers `A` through `F`.
2. The exit is marked near the final path endpoint.
3. Closeness to the exit is path order, not straight-line distance.
4. The answer is the label of the enemy farthest along the path toward the exit.
5. Annotation is the point at the center of the selected labeled enemy.
