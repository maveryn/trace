# `task_games__rhythm__lane_note_count`

## Program Contract

Program: `count(filter(notes, lane=target_lane)); scene=rhythm; scope=lane_note_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `lane_note_count` objective scope.
Operands: visible scene state and prompt-bound target operands named by the task contract.
Operation: evaluate `count(filter(notes, lane=target_lane))` over visible rhythm notes; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema for all note objects in the target lane.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Notes
- `target_lane` is prompt-bound by lane number.
- A long vertical note counts as one note object.
- Annotation bboxes are all note objects in the specified lane.
- Scalar annotation checked: true.
