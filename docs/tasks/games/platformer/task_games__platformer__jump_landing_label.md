# `task_games__platformer__jump_landing_label`

## Contract
1. Domain: `games`
2. Scene id: `platformer`
3. Public task id: `task_games__platformer__jump_landing_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string_label`
6. Annotation schema: `bbox`
7. Program schema: `label(landing_platform(marked_jump)); scene=platformer; scope=jump_landing_label`

## Program Contract

Program: `scene=platformer; scope=jump_landing_label; program=label(landing_platform(shown_jump_arc))`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `jump_landing_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `platformer`, `jump_landing_label`, `program`, `label`, `landing_platform`, `shown_jump_arc`.
Operation: evaluate `scene=platformer` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
