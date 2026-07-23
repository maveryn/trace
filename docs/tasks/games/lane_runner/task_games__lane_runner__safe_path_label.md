# `task_games__lane_runner__safe_path_label`

## Contract
1. Domain: `games`
2. Scene id: `lane_runner`
3. Public task id: `task_games__lane_runner__safe_path_label`
4. Supported `query_id` values: `single`
5. Annotation schema: `bbox`

## Program Contract

Program: `select_unique(label(path) where no_hazard_collision(path, hazards)); scene=lane_runner; scope=safe_path_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `safe_path_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `path`, `where`, `no_hazard_collision`, `hazards`, `lane_runner`, `safe_path_label`.
Operation: evaluate `select_unique` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Answer And Annotation
1. `answer_gt.type`: `option_letter`.
2. `annotation_gt.type`: `bbox`.
3. Annotation is the bounding box around the selected path card.
4. The sampler uses only four-option or six-option sets and rejects instances unless exactly one displayed path avoids all hazards.
