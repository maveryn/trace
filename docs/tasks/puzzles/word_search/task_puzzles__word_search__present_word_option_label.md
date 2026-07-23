# `task_puzzles__word_search__present_word_option_label`

## Program Contract

Program: `select_label(word_option, rule=option_word_appears_in_grid); scene=word_search; scope=present_word_option_label`

Candidate set: the visible letter grid, target or option words, candidate location/direction labels, and highlighted word path inside the `present_word_option_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `word_option`, `option_word_appears_in_grid`, `word_search`, `present_word_option_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the option label for the word present in the grid.
Annotation witnesses: `annotation` uses the `segment` schema; one image-pixel segment from the present word's first-letter.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `matching`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the option label for the word present in the grid.
3. `annotation_gt.type = segment`
4. Annotation schema: `segment`
5. Annotation is one image-pixel segment from the present word's first-letter
   cell center to its last-letter cell center.
6. `scalar_annotation_checked = true`; this task has a single visual witness.

## Query Contract
1. Public `query_id`: `single`
2. Option labels are visible answer candidates, not public query branches.
3. Internal variation covers option words, target word placement, grid size, and
   scene styling only.

## Prompt Contract
1. Bundle: `puzzles_word_search_v1`
2. Scene key: `word_search`
3. Task key: `present_word_option_label_query`
4. Prompt query key: `present_word_option_label`
