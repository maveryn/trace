# `task_puzzles__cube_net__marked_edge_neighbor_face_label`

## Summary
1. Domain: `puzzles`
2. Scene id: `cube_net`
3. Task id: `task_puzzles__cube_net__marked_edge_neighbor_face_label`
4. Objective contract: choose the option matching the face across the red marked edge after folding.

## Program Contract

Program: `select_option(folded_cube_net.edge_neighbor_face, edge=marked_edge); scene=cube_net; scope=marked_edge_neighbor_face_label`

Candidate set: the visible cube-net faces, face colors/labels, marked face or edge, and labeled candidate options inside the `marked_edge_neighbor_face_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `folded_cube_net`, `edge_neighbor_face`, `edge`, `marked_edge`, `cube_net`, `marked_edge_neighbor_face_label`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `transformation`

## Query Contract
1. Public query id: `single`
2. The task always asks for the folded-cube face that shares the red marked edge.
3. The marked edge is sampled from exposed flat-net edges so it belongs unambiguously to one visible reference face before folding.
4. Net rotation, style, face labels, and option order are render/generation axes, not query ids.

## Answer And Annotation
1. Answer type: `option_letter`
2. Annotation schema: `bbox`
3. Annotation box is the image-pixel box for the selected option card.
4. The red marked edge is already visible in the image and is not part of the annotation.

## Implementation
1. Registered class: `trace_tasks.tasks.puzzles.cube_net.marked_edge_neighbor_face_label.PuzzlesCubeNetMarkedEdgeNeighborFaceLabelTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/puzzles/cube_net/puzzles_cube_net_v1.json`
3. Scene config: `src/trace_tasks/resources/configs/domains/puzzles/cube_net.yaml`
