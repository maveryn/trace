# `task_games__minecraft__stack_height_condition_count`

## Contract
1. Domain: `games`
2. Scene id: `minecraft`
3. Public task id: `task_games__minecraft__stack_height_condition_count`
4. Supported `query_id` values: `exact_height_count`, `at_least_height_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(stacks, height_relation(stack.height, target_height)=exact|at_least)); scene=minecraft; scope=stack_height_condition_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `stack_height_condition_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `stacks`, `height_relation`, `stack`, `height`, `target_height`, `exact`, `at_least`, `minecraft`, `stack_height_condition_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `exact_height_count`, `at_least_height_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Generation Notes
1. The scene contains visible cube columns with contiguous block levels from the ground upward.
2. `exact_height_count` asks for stacks exactly the target height.
3. `at_least_height_count` asks for stacks at least the target height.
4. Annotation boxes enclose every qualifying visible stack.
5. The generated answer support is capped at 5.
