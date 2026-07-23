# `task_games__slot_machine__reel_completion_label`

## Contract
1. Domain: `games`
2. Scene: `slot_machine`
3. Scene id: `slot_machine`
4. Public task id: `task_games__slot_machine__reel_completion_label`
5. Supported `query_id` values: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `bbox`

## Program Contract

Program: `unique_label(option for option in third_reel_options if count(payline for payline in rows_plus_long_diagonals if all_symbols_match(first_two_reels + option)) == 1); scene=slot_machine; scope=reel_completion_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `reel_completion_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `third_reel_options`, `if`, `payline`, `rows_plus_long_diagonals`, `all_symbols_match`, `first_two_reels`, `slot_machine`, `reel_completion_label`.
Operation: evaluate `unique_label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `formula_evaluation`, `matching`

## Generation Notes
1. The scene renders the first two visible reels of a 3x3 slot machine and four labeled candidate third reels.
2. Paylines are the three full rows plus the two long diagonals; columns are not paylines.
3. Exactly one candidate third reel completes one matching-symbol payline.
4. The other three candidate third reels complete no paylines.
5. The answer is the selected option letter.
6. Annotation is the scalar bounding box around the selected third-reel option panel.
7. Scalar annotation checked: true.
