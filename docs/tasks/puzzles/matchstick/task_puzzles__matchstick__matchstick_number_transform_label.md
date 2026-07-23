# `task_puzzles__matchstick__matchstick_number_transform_label`

## Program Contract

Program: `select_option(matchstick_number.one_stick_transform, operation=add_one|remove_one, source=visible_source_number); scene=matchstick; scope=source_number_and_candidate_options`

Candidate set: the visible matchstick segments, digit/equation/lattice structure, segment labels, and labeled candidate options when present inside the `source_number_and_candidate_options` objective scope.
Operands: visible scene state and prompt-bound operands named by `matchstick_number`, `one_stick_transform`, `operation`, `add_one`, `remove_one`, `source`, `visible_source_number`, `matchstick`, `source_number_and_candidate_options`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `state_update`

## Answer And Annotation

- Answer type: `option_letter`.
- Annotation type: `bbox_map`.
- Annotation schema: `bbox_map`.
- Annotation keys:
  - `source_number`: bbox around the Source panel.
  - `selected_option`: bbox around the selected candidate panel.
- The answer and annotation are bound from the same sampled trace. The trace records the source number, answer number, changed digit index, added/removed segment keys, and per-option reachability.

## Rendering And Prompt

The `matchstick` scene renders one Source panel and six labeled candidate panels using wooden-match, colored-rod, chalk-stick, neon-rod, or metal-rod visual styles. Prompt prose comes from `src/trace_tasks/resources/prompts/puzzles/matchstick/puzzles_matchstick_v1.json`.
