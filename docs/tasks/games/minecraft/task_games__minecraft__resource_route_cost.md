# `task_games__minecraft__resource_route_cost`

## Contract
1. Domain: `games`
2. Scene id: `minecraft`
3. Public task id: `task_games__minecraft__resource_route_cost`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(track_cells, has_raised_stone_or_dirt_block=true)); scene=minecraft; scope=resource_route_cost`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `resource_route_cost` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `track_cells`, `has_raised_stone_or_dirt_block`, `true`, `minecraft`, `resource_route_cost`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Generation Notes
1. The scene shows one visible track across an isometric block world.
2. Each raised stone or dirt block sitting on the track counts 1.
3. Empty track cells and raised blocks away from the track do not count.
4. Annotation boxes enclose every counted raised block on the track.
5. Counted raised blocks are never placed on the first or last track cell.
6. Off-track distractor blocks are sampled at least 3 grid cells away from the visible track.
