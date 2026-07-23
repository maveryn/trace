# `task_games__solitaire__move_legality_label`

## Contract
1. Domain: `games`
2. Scene id: `solitaire`
3. Public task id: `task_games__solitaire__move_legality_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox_map`
7. Program schema: `select_option(legal_solitaire_move); scene=solitaire; scope=move_legality_label`
8. Scalar annotation checked: `true`

## Program Contract

Program: `select_option(legal_solitaire_move); scene=solitaire; scope=move_legality_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `move_legality_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `legal_solitaire_move`, `solitaire`, `move_legality_label`.
Operation: evaluate `select_option` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. The scene shows tableau columns, four foundation piles, and move options.
2. Exactly four visible options are shown, using `Col N -> target` text, and exactly one option is legal by solitaire tableau/foundation rules.
3. Annotation is a bbox map with `source_card` and `target` roles for the legal move.
4. Prompt wording comes from `src/trace_tasks/resources/prompts/games/solitaire/games_solitaire_v1.json`.
