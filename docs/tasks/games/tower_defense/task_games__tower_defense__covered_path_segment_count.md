# `task_games__tower_defense__covered_path_segment_count`

## Contract
1. Domain: `games`
2. Scene: `tower_defense`
3. Scene id: `tower_defense`
4. Public task id: `task_games__tower_defense__covered_path_segment_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `point_set`

## Program Contract

Program: `count(path_enemy for path_enemy in path_enemies if any(path_enemy_center inside tower_range_circle for tower in towers)); scene=tower_defense; scope=covered_path_segment_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `covered_path_segment_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `path_enemy`, `path_enemies`, `if`, `any`, `path_enemy_center`, `inside`, `tower_range_circle`, `tower`, `towers`, `tower_defense`, `covered_path_segment_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `spatial_relations`, `topology`

## Generation Notes
1. The path is drawn with small visible enemy markers along a winding or switchback route.
2. Towers are placed off the path and display circular range rings.
3. A path enemy is covered when its center lies inside at least one tower range ring.
4. Annotation contains one point at the center of every covered path enemy.
