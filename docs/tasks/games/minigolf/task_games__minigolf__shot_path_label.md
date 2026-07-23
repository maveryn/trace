# `task_games__minigolf__shot_path_label`

## Contract
1. Domain: `games`
2. Scene id: `minigolf`
3. Public task id: `task_games__minigolf__shot_path_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string_label`
6. Annotation schema: `segment`

## Program Contract

Program: `label(select_option(shot_paths, option_rule=path_satisfies_target)); scene=minigolf; scope=shot_path_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `shot_path_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select_option`, `shot_paths`, `option_rule`, `path_satisfies_target`, `minigolf`, `shot_path_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`, `topology`, `state_update`

## Generation Notes
1. Query ids are internal replay/sampling keys and do not define public task units.
2. Annotation is projected from the same generated game state used for answer verification.
