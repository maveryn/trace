# `task_games__pacman__pellet_count_before_ghost`

## Contract
1. Domain: `games`
2. Scene id: `pacman`
3. Public task id: `task_games__pacman__pellet_count_before_ghost`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `point_set_map`

## Program Contract

Program: `count(filter(prefix(route_cells, before=first(route_cell where contains_ghost=True)), contains_normal_pellet=True)); scene=pacman; scope=pellet_count_before_ghost`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `pellet_count_before_ghost` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `prefix`, `route_cells`, `before`, `first`, `route_cell`, `where`, `contains_ghost`, `True`, `contains_normal_pellet`, `pacman`, `pellet_count_before_ghost`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `spatial_relations`, `topology`

## Generation Notes
1. The highlighted route starts at the visible Pac-Man marker.
2. The answer counts normal pellets before the first ghost encountered on that highlighted route.
3. Annotation uses `counted_pellets` for counted pellet center points and `first_ghost` for the stopping ghost center point.
