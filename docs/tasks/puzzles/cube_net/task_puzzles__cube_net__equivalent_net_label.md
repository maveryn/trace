# `task_puzzles__cube_net__equivalent_net_label`

## Summary
1. Domain: `puzzles`
2. Scene id: `cube_net`
3. Task id: `task_puzzles__cube_net__equivalent_net_label`
4. Objective contract: choose the candidate net that folds into the same colored cube as the reference net.

## Program Contract

Program: `select_option(cube_net_equivalence.matching_colored_cube, reference=colored_reference_net, equivalence=whole_cube_rotation); scene=cube_net; scope=equivalent_net_label`

Candidate set: the visible cube-net faces, face colors/labels, marked face or edge, and labeled candidate options inside the `equivalent_net_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `cube_net_equivalence`, `matching_colored_cube`, `reference`, `colored_reference_net`, `equivalence`, `whole_cube_rotation`, `cube_net`, `equivalent_net_label`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Contract
1. Public query id: `single`
2. The task always asks for the option net equivalent to the reference colored net.
3. Color assignment, option order, net display rotation, and style are generation/render axes, not query ids.

## Answer And Annotation
1. Answer type: `option_letter`
2. Annotation schema: `bbox`
3. Annotation box marks only the selected option net panel.
4. The reference net is already visible in the image and is not part of the annotation.

## Implementation
1. Registered class: `trace_tasks.tasks.puzzles.cube_net.equivalent_net_label.PuzzlesCubeNetEquivalentNetLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/puzzles/cube_net/puzzles_cube_net_v1.json`
3. Scene config: `src/trace_tasks/resources/configs/domains/puzzles/cube_net.yaml`
