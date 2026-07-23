# `task_puzzles__matchstick__equation_repair_stick_label`

## Program Contract

Program: `select_option(matchstick_equation.remove_one_stick_repair, source=false_visible_equation, candidates=labeled_digit_sticks); scene=matchstick; scope=false_equation_labeled_digit_sticks`

Candidate set: the visible matchstick segments, digit/equation/lattice structure, segment labels, and labeled candidate options when present inside the `false_equation_labeled_digit_sticks` objective scope.
Operands: visible scene state and prompt-bound operands named by `matchstick_equation`, `remove_one_stick_repair`, `source`, `false_visible_equation`, `candidates`, `labeled_digit_sticks`, `matchstick`, `false_equation_labeled_digit_sticks`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`, `formula_evaluation`, `matching`

## Answer And Annotation

- Answer type: `option_letter`.
- Annotation type: `segment`.
- Annotation schema: `segment`.
- Annotation value: `[[x1, y1], [x2, y2]]`, the image-pixel centerline endpoints of the selected labeled stick.
- The answer and annotation are bound from the same sampled trace. The trace records every possible digit-stick removal and guarantees exactly one removal repairs the equation.

## Rendering And Prompt

The `matchstick` scene renders a single false matchstick equation in one panel. Four or six candidate digit sticks are labeled directly on the equation. Prompt prose comes from `src/trace_tasks/resources/prompts/puzzles/matchstick/puzzles_matchstick_v1.json`.
