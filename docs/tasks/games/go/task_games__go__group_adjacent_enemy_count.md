# `task_games__go__group_adjacent_enemy_count`

## Contract
1. Domain: `games`
2. Scene: `go`
2. Scene id: `go`
3. Public task id: `task_games__go__group_adjacent_enemy_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(filter(stones, adjacent_to_group(stone, marked_group) and stone_color=opponent_color)); scene=go; scope=group_adjacent_enemy_count`

## Program Contract

Program: `count(filter(stones, adjacent_to_group(stone, marked_group) and stone_color=opponent_color)); scene=go; scope=group_adjacent_enemy_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `group_adjacent_enemy_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `stones`, `adjacent_to_group`, `stone`, `marked_group`, `stone_color`, `opponent_color`, `go`, `group_adjacent_enemy_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `spatial_relations`, `topology`

## Generation Notes
1. The task samples a marked black or white connected group and asks for opponent stones touching it orthogonally.
2. `query_id=single` is the public no-branch query id; the prompt query key remains the semantic prompt template.
3. Annotation marks the opponent-stone boxes directly adjacent to the marked group.
