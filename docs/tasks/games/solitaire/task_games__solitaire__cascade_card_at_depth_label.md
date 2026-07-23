# `task_games__solitaire__cascade_card_at_depth_label`

## Contract
1. Domain: `games`
2. Scene id: `solitaire`
3. Public task id: `task_games__solitaire__cascade_card_at_depth_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `point`
7. Program schema: `select_option(card_at_visible_depth(column, depth)); scene=solitaire; scope=cascade_card_at_depth_label`
8. Scalar annotation checked: `true`

## Program Contract

Program: `select_option(card_at_visible_depth(column, depth)); scene=solitaire; scope=cascade_card_at_depth_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `cascade_card_at_depth_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `card_at_visible_depth`, `column`, `depth`, `solitaire`, `cascade_card_at_depth_label`.
Operation: evaluate `select_option` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Generation Notes
1. The prompt names a 1-based tableau column and a visible depth counted from the top of that column.
2. The visual options show card faces, and exactly one option matches the target tableau card.
3. Annotation is one point on the visible part of the target card in the tableau, not the option card.
4. Prompt wording comes from `src/trace_tasks/resources/prompts/games/solitaire/games_solitaire_v1.json`.
