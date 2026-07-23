# `task_games__bubble_shooter__pop_target_label`

## Contract
1. Domain: `games`
2. Scene id: `bubble_shooter`
3. Public task id: `task_games__bubble_shooter__pop_target_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string_label`
6. Annotation schema: `bbox`
7. Program schema: `label(unique_landing_target_where(count(existing_same_color_component_adjacent_to(target, shooter_color)) + 1 >= 3)); scene=bubble_shooter; scope=pop_target_label`

## Program Contract

Program: `label(unique_landing_target_where(count(existing_same_color_component_adjacent_to(target, shooter_color)) + 1 >= 3)); scene=bubble_shooter; scope=pop_target_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `pop_target_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `unique_landing_target_where`, `existing_same_color_component_adjacent_to`, `shooter_color`, `bubble_shooter`, `pop_target_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `topology`, `state_update`

## Generation Notes
1. This task is owned by the current-layout public file `src/trace_tasks/tasks/games/bubble_shooter/pop_target_label.py`.
2. The public task id selects the objective; `query_id` is `single`.
3. Annotation is projected from the same generated game state used for answer verification.
