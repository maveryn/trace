# `task_games__darts__bullseye_membership_count`

## Contract
1. Domain: `games`
2. Scene: `darts`
3. Scene id: `darts`
4. Public task id: `task_games__darts__bullseye_membership_count`
5. Supported `query_id` values: `inside_bullseye_count`, `outside_bullseye_count`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(filter(darts, bullseye_membership(dart)=inside|outside)); scene=darts; scope=bullseye_membership_count`

## Program Contract

Program: `count(filter(darts, bullseye_membership(dart)=inside|outside)); scene=darts; scope=bullseye_membership_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `bullseye_membership_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `darts`, `bullseye_membership`, `dart`, `inside`, `outside`, `bullseye_membership_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `inside_bullseye_count`, `outside_bullseye_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Generation Notes
1. The scene renders a simplified dartboard with 10 numbered sectors and one center bullseye.
2. Query ids switch only the user-facing membership predicate: inside vs outside the bullseye.
3. The answer support is `0..5`.
4. Annotation marks the dart boxes whose bullseye membership matches the query.
