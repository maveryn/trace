# `task_puzzles__word_search__search_location_label`

## Program Contract

Program: `select_label(location_option, rule=target_word_start_cell_and_direction); scene=word_search; scope=search_location_label`

Candidate set: the visible letter grid, target or option words, candidate location/direction labels, and highlighted word path inside the `search_location_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `location_option`, `target_word_start_cell_and_direction`, `word_search`, `search_location_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the correct option label.
Annotation witnesses: `annotation` uses the `bbox_sequence` schema; the ordered sequence of grid-cell bounding boxes for the found.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`, `topology`, `matching`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the correct option label.
3. `annotation_gt.type = bbox_sequence`
4. Annotation schema: `bbox_sequence`
5. Annotation is the ordered sequence of grid-cell bounding boxes for the found
   word, from first letter to last letter.
6. `scalar_annotation_checked = true`; this task has multiple ordered cell
   witnesses.

## Query Contract
1. Public `query_id`: `single`
2. Option labels are visible answer candidates, not public query branches.
3. Internal variation covers word, direction, grid size, and scene styling only.

## Prompt Contract
1. Bundle: `puzzles_word_search_v1`
2. Scene key: `word_search`
3. Task key: `search_location_label_query`
4. Prompt query key: `search_location_label`
