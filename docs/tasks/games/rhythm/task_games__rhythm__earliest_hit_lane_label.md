# `task_games__rhythm__earliest_hit_lane_label`

## Program Contract

Program: `argmin(filter(notes, note_in_hit_window(note, beat_window)=true), metric=bottom_row_from_hit_line).lane_label; scene=rhythm; scope=earliest_hit_lane_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `earliest_hit_lane_label` objective scope.
Operands: visible scene state and prompt-bound target operands named by the task contract.
Operation: evaluate `argmin(filter(notes, note_in_hit_window(note, beat_window)=true), metric=bottom_row_from_hit_line).lane_label` over visible rhythm notes; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the value is the one-based visible lane number containing the uniquely earliest hitting note.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `comparison`, `ranking`

## Notes
- The sampled scene has one uniquely earliest hitting note by construction.
- Annotation is the scalar bbox around that earliest note.
- Scalar annotation checked: true.
