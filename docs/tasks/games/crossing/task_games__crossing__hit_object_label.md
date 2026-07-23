# `task_games__crossing__hit_object_label`

## Contract
1. Domain: `games`
2. Scene id: `crossing`
3. Public task id: `task_games__crossing__hit_object_label`
4. Supported `query_id` values: `single`
5. Answer schema: `label_string`
6. Annotation schema: `point`
7. Program schema: `label(collision(marked_route, labeled_moving_objects)); scene=crossing; scope=hit_object_label`

## Program Contract

Program: `label(collision(marked_route, labeled_moving_objects)); scene=crossing; scope=hit_object_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `hit_object_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `collision`, `marked_route`, `labeled_moving_objects`, `crossing`, `hit_object_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`, `topology`

## Generation Notes
1. Exactly four moving objects are labeled `A` through `D`; the answer is one of those labels.
2. Start pads use numeric labels so they do not conflict with moving-object option labels.
3. Annotation is the center point of the single labeled moving object that reaches the marked route at the matching tick.
