# `task_games__darts__highest_scoring_dart_label`

## Contract
1. Domain: `games`
2. Scene: `darts`
3. Scene id: `darts`
4. Public task id: `task_games__darts__highest_scoring_dart_label`
5. Supported `query_id` values: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `point`
8. Program schema: `argmax(label for labeled_dart in darts, score(labeled_dart)); scene=darts; scope=highest_scoring_dart_label`

## Program Contract

Program: `argmax(label for labeled_dart in darts, score(labeled_dart)); scene=darts; scope=highest_scoring_dart_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `highest_scoring_dart_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `labeled_dart`, `score`, `darts`, and `highest_scoring_dart_label`.
Operation: evaluate `argmax` over the labeled dart set using the simplified dartboard scoring rule; generation enforces a unique highest-scoring labeled dart.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema at the center of the selected dart.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Generation Notes
1. The scene renders a simplified dartboard with 10 numbered sectors, one center bullseye, and four visible darts labeled `A` through `D`.
2. A dart in a numbered sector scores that number; a dart in the center bullseye scores `50`.
3. Generated instances place the four labeled darts in numbered sectors and enforce one unique highest score.
4. Query ids are internal replay/sampling keys and do not define public task units.
5. Annotation is the center point of the highest-scoring labeled dart, projected from the same generated game state used for answer verification.
