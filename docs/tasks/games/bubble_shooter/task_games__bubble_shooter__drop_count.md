# `task_games__bubble_shooter__drop_count`

## Contract
1. Domain: `games`
2. Scene id: `bubble_shooter`
3. Public task id: `task_games__bubble_shooter__drop_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(disconnected_bubbles_after_pop(marked_shot)); scene=bubble_shooter; scope=drop_count`

## Program Contract

Program: `count(disconnected_bubbles_after_pop(marked_shot)); scene=bubble_shooter; scope=drop_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `drop_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `disconnected_bubbles_after_pop`, `marked_shot`, `bubble_shooter`, `drop_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `topology`, `state_update`

## Generation Notes
1. This task is owned by the current-layout public file `src/trace_tasks/tasks/games/bubble_shooter/drop_count.py`.
2. The public task id selects the objective; `query_id` is `single`.
3. Annotation is projected from the same generated game state used for answer verification.
