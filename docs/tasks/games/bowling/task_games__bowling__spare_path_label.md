# `task_games__bowling__spare_path_label`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/bowling/`
3. Scene id: `bowling`
4. Public task id: `task_games__bowling__spare_path_label`
5. Supported `query_id` values: `single`
6. Answer schema: `string_label`
7. Annotation schema: `segment`
8. Program schema: `label(select_option(shot_paths, option_rule=clears_remaining_pins)); scene=bowling; scope=spare_path_label`

## Program Contract

Program: `label(select_option(shot_paths, option_rule=clears_remaining_pins)); scene=bowling; scope=spare_path_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `spare_path_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select_option`, `shot_paths`, `option_rule`, `clears_remaining_pins`, `bowling`, `spare_path_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`, `topology`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
4. Annotation is one `segment` `[[x0, y0], [x1, y1]]` using the endpoints of the selected visible dashed cue.
