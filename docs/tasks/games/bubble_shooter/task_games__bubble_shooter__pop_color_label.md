# `task_games__bubble_shooter__pop_color_label`

## Contract
1. Domain: `games`
2. Scene id: `bubble_shooter`
3. Public task id: `task_games__bubble_shooter__pop_color_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string_label`
6. Annotation schema: `bbox_set`
7. Program schema: `label(color(inserted_group(marked_shot))); scene=bubble_shooter; scope=pop_color_label`

## Program Contract

Program: `label(color(inserted_group(marked_shot))); scene=bubble_shooter; scope=pop_color_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `pop_color_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `color`, `inserted_group`, `marked_shot`, `bubble_shooter`, `pop_color_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. This task is owned by the current-layout public file `src/trace_tasks/tasks/games/bubble_shooter/pop_color_label.py`.
2. The public task id selects the objective; `query_id` is `single`.
3. Annotation is projected from the same generated game state used for answer verification.
