# `task_games__crossing__first_exit_object_label`

## Contract
1. Domain: `games`
2. Scene id: `crossing`
3. Public task id: `task_games__crossing__first_exit_object_label`
4. Supported `query_id` values: `single`
5. Answer schema: `label_string`
6. Annotation schema: `point`
7. Program schema: `label(argmin(labeled_moving_objects, exit_tick(object, lane_count, direction))); scene=crossing; scope=first_exit_object_label`

## Program Contract

Program: `label(argmin(labeled_moving_objects, exit_tick(object, lane_count, direction))); scene=crossing; scope=first_exit_object_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `first_exit_object_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `argmin`, `labeled_moving_objects`, `exit_tick`, `object`, `lane_count`, `direction`, `crossing`, `first_exit_object_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`

## Generation Notes
1. Exactly four moving objects are labeled `A` through `D`; the answer is one of those labels.
2. The scene has no runner route; each labeled object moves horizontally by one lane cell per tick.
3. The target labeled object is the unique labeled object that exits past a left or right edge first.
4. Annotation is the center point of the labeled moving object that leaves the board first.
