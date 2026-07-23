# `task_puzzles__cube_net__opposite_face_label`

## Summary
1. Domain: `puzzles`
2. Scene id: `cube_net`
3. Task id: `task_puzzles__cube_net__opposite_face_label`
4. Objective contract: choose the option matching the face opposite the marked face after folding.

## Program Contract

Program: `select_option(folded_cube_net.opposite_face, reference=marked_face); scene=cube_net; scope=opposite_face_label`

Candidate set: the visible cube-net faces, face colors/labels, marked face or edge, and labeled candidate options inside the `opposite_face_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `folded_cube_net`, `opposite_face`, `reference`, `marked_face`, `cube_net`, `opposite_face_label`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## Query Contract
1. Public query id: `single`
2. The task always asks for the folded-cube face opposite the marked reference face.
3. Net rotation, style, face labels, and option order are render/generation axes, not query ids.

## Answer And Annotation
1. Answer type: `option_letter`
2. Annotation schema: `bbox`
3. Annotation box is the image-pixel box for the selected option card.
4. The marked reference face is already visible in the image and is not part of the annotation.

## Implementation
1. Registered class: `trace_tasks.tasks.puzzles.cube_net.opposite_face_label.PuzzlesCubeNetOppositeFaceLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/puzzles/cube_net/puzzles_cube_net_v1.json`
3. Scene config: `src/trace_tasks/resources/configs/domains/puzzles/cube_net.yaml`
