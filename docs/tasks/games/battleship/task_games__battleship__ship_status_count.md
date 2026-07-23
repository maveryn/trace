# `task_games__battleship__ship_status_count`

## Contract
1. Domain: `games`
2. Scene id: `battleship`
3. Public task id: `task_games__battleship__ship_status_count`
4. Supported `query_id` values: `sunk_ship_count`, `partial_ship_count`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set_map`
7. Program schema: `count(filter(ships, ship_status=target_status)); scene=battleship; scope=ship_status_count; query_branch=partial_ship_count`

## Program Contract

Program: `count(filter(ships, ship_status=target_status)); scene=battleship; scope=ship_status_count; query_branch=partial_ship_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `ship_status_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `ships`, `ship_status`, `target_status`, `battleship`, `ship_status_count`, `query_branch`, `partial_ship_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `sunk_ship_count`, `partial_ship_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation maps each counted ship name to the hit-cell boxes on that ship, projected from the same generated game state used for answer verification.
