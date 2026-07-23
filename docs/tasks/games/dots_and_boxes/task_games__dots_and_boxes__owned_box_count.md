# `task_games__dots_and_boxes__owned_box_count`

## Contract
1. Domain: `games`
2. Scene id: `dots_and_boxes`
3. Public task id: `task_games__dots_and_boxes__owned_box_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(filter(completed_boxes, owner_mark(box)=queried_player)); scene=dots_and_boxes; scope=owned_box_count`

## Program Contract

Program: `count(filter(completed_boxes, owner_mark(box)=queried_player)); scene=dots_and_boxes; scope=owned_box_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `owned_box_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `completed_boxes`, `owner_mark`, `box`, `queried_player`, `dots_and_boxes`, `owned_box_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. The scene renders a visible dots-and-boxes board with completed boxes marked by player `A` or player `B`.
2. The target owner mark is sampled with `target_owner` (`A` or `B`).
3. Annotation is projected from the same generated game state used for answer verification.
