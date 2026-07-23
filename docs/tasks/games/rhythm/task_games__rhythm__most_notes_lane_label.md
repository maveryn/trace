# `task_games__rhythm__most_notes_lane_label`

## Program Contract

Program: `argmax(lanes, metric=count(filter(notes, lane=lane))).label; scene=rhythm; scope=most_notes_lane_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `most_notes_lane_label` objective scope.
Operands: visible scene state and prompt-bound target operands named by the task contract.
Operation: evaluate `argmax(lanes, metric=count(filter(notes, lane=lane))).label` over visible rhythm notes; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the value is the one-based visible lane number with the unique maximum note-object count.
Annotation witnesses: `annotation` uses the `bbox_set` schema for all note objects in the winning lane.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`

## Notes
- The sampled scene has one lane with a unique maximum note-object count.
- A long vertical note counts as one note object.
- Annotation bboxes are all note objects in the winning lane.
- Scalar annotation checked: true.
