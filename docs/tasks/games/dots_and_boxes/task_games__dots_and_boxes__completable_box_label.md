# `task_games__dots_and_boxes__completable_box_label`

## Contract
1. Domain: `games`
2. Scene id: `dots_and_boxes`
3. Public task id: `task_games__dots_and_boxes__completable_box_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`
7. Program schema: `label(filter(candidate_boxes, drawn_side_count(box)=3)); scene=dots_and_boxes; scope=completable_box_label`

## Program Contract

Program: `label(filter(candidate_boxes, drawn_side_count(box)=3)); scene=dots_and_boxes; scope=completable_box_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `completable_box_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `candidate_boxes`, `drawn_side_count`, `box`, `dots_and_boxes`, `completable_box_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`

## Generation Notes
1. The scene renders exactly six labeled box options `A` through `F`.
2. Exactly one labeled option box has exactly three drawn sides and can be completed by drawing one missing side.
3. Query ids are internal replay/sampling keys and do not define public task units.
4. Annotation is the full-cell bounding box of the selected labeled box, projected from the same generated game state used for answer verification.
