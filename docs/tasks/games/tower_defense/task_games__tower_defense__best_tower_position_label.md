# `task_games__tower_defense__best_tower_position_label`

## Contract
1. Domain: `games`
2. Scene: `tower_defense`
3. Scene id: `tower_defense`
4. Public task id: `task_games__tower_defense__best_tower_position_label`
5. Supported `query_id` values: `single`
6. Answer schema: `label`
7. Annotation schema: `point`

## Program Contract

Program: `argmax_label(candidate in candidates_A_to_D, count(path_enemy for path_enemy in path_enemies if path_enemy_center inside candidate_range_circle)); scene=tower_defense; scope=best_tower_position_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `best_tower_position_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `candidate`, `candidates_A_to_D`, `path_enemy`, `path_enemies`, `if`, `path_enemy_center`, `inside`, `candidate_range_circle`, `tower_defense`, `best_tower_position_label`.
Operation: evaluate `argmax_label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `spatial_relations`, `topology`, `state_update`

## Generation Notes
1. The path is drawn with small visible enemy markers along a winding or switchback route.
2. Four candidate tower positions are labeled `A`, `B`, `C`, and `D`, each with a circular range ring.
3. All four candidate tower range rings use the same radius within an instance.
4. A candidate covers a path enemy when the enemy center lies inside that candidate's range ring.
5. Exactly one candidate covers the most path enemies by construction.
6. Annotation is the point at the center of the selected candidate tower.
