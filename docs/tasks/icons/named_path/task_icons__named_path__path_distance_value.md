# `task_icons__named_path__path_distance_value`

- domain: `icons`
- scene_id: `named_path`
- task: `path_distance_value`
- module: `src/trace_tasks/tasks/icons/named_path/path_distance_value.py`

## Program Contract

Program: `count.path_stops_between_named_icons(scene=named_path, scope=ordered_path_stops, endpoints=two_distinct_icon_types, output=integer)`

Candidate set: visible path stops along the single START-to-END icon path.
Operands: two prompt-bound named icon types, each appearing exactly once in the
rendered path.
Operation: find the two endpoint icon occurrences and count the path stops
strictly between them by START-to-END order.
Output binding: `answer` uses the `integer` schema; generation binds a unique
count by construction.
Annotation witnesses: `annotation` uses the `bbox_set` schema for every counted
path-stop icon strictly between the two prompt-bound endpoint icons.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Contract
1. The image shows a single continuous open path marked from `START` to `END`.
2. The two prompt-bound named icon types each appear exactly once.
3. Other path stops use non-endpoint icon types.
4. The answer is the number of path stops strictly between the two named icons.
5. `answer_gt.type = integer`.
6. `annotation_gt.type = bbox_set` with one bbox for each counted intermediate
   path-stop icon.

## Query IDs
- `single`

## Generation
Default answer support is `1..8`. The generator adds `2..6` extra stops outside
the two endpoint icons so the answer cannot be inferred from total stop count
alone.

The endpoint icon types are sampled from the procedural named-icon vocabulary.
Endpoint order along the path is an internal generation axis and does not change
the public query contract.

## Trace
The trace records:
- ordered path points in pixel coordinates,
- every icon stop with path position, shape, color, fill style, bbox, and role,
- the two endpoint icon positions and names,
- all positions strictly between the endpoints,
- render-style metadata for path and text legibility,
- answer, endpoint-shape, extra-stop, and fill-style probability metadata.
