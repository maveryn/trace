# `task_icons__named_path__path_neighbor_label`

- domain: `icons`
- scene_id: `named_path`
- task: `path_neighbor_label`
- module: `src/trace_tasks/tasks/icons/named_path/path_neighbor_label.py`

## Program Contract

Program: `selection.path_neighbor(scene=named_path, scope=ordered_path_stops, reference_occurrence=first|second|last, direction=before|after, output=option_letter)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `ordered_path_stops` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_path`, `ordered_path_stops`, `reference_occurrence`, `first`, `second`, `last`, `direction`, `before`, `after`.
Operation: evaluate `selection.path_neighbor` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `after_first_shape_label`, `before_first_shape_label`,
`after_last_shape_label`, `before_last_shape_label`,
`after_second_shape_label`, `before_second_shape_label`.

## Reasoning Operations

Families: `ranking`, `topology`

## Contract
1. The image shows a single continuous open path marked from `START` to `END`.
2. Procedural named icons are placed on ordered path stops.
3. Exactly six non-target stop icons are option icons labeled `A` through `F`; the answer is one option letter.
4. The queried named icon occurrence is selected by path order and is not one of the option icons.
5. `answer_gt.type = option_letter`.
6. `annotation_gt.type = bbox` for the selected labeled answer icon.
   Annotation marks the answer icon itself, not the queried named-icon
   occurrence, numeric path stops, or standalone label text.

## Query IDs
- `after_first_shape_label`
- `before_first_shape_label`
- `after_last_shape_label`
- `before_last_shape_label`
- `after_second_shape_label`
- `before_second_shape_label`

## Generation
The target shape is sampled from the full procedural named-icon vocabulary in
`src/trace_tasks/tasks/icons/shared/procedural_named_icons.py`. The path contains six
labeled option stops, `4..8` other non-target stops, and `2..4`
occurrences of the target shape. Target occurrences are non-adjacent and never
placed on the path endpoints. The selected neighbor is always a labeled
non-target option.

Prompt text quotes the target shape name, for example `"guitar"` icons.

## Trace
The trace records:
- ordered path points in pixel coordinates,
- every icon stop with `position_index`, label, shape, color, fill style, bbox,
  and target-occurrence metadata,
- the queried target occurrence,
- the selected answer option,
- render-style metadata for panel title, option-label, and START/END text
  legibility,
- query probabilities and sampled support metadata.
