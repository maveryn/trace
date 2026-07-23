# `task_games__minecraft__top_ore_stack_count`

## Contract
1. Domain: `games`
2. Scene id: `minecraft`
3. Public task id: `task_games__minecraft__top_ore_stack_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(stacks, top_block_type=target_ore_type)); scene=minecraft; scope=top_ore_stack_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `top_ore_stack_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `stacks`, `top_block_type`, `target_ore_type`, `minecraft`, `top_ore_stack_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. The scene shows visible cube stacks in an isometric Minecraft-like block world.
2. The target ore type is sampled from gold ore or diamond ore.
3. The answer counts stacks whose top cube is the named ore, not ore blocks hidden below the top.
4. Annotation boxes enclose every counted stack.
