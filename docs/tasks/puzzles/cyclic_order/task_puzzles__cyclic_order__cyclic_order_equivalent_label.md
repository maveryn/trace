# `task_puzzles__cyclic_order__cyclic_order_equivalent_label`

## Contract

1. Domain: `puzzles`
2. Scene package: `src/trace_tasks/tasks/puzzles/cyclic_order/`
3. Scene id: `cyclic_order`
4. Public task id: `task_puzzles__cyclic_order__cyclic_order_equivalent_label`
5. Supported `query_id` values: `single`
6. Prompt query key: `single`
7. Answer schema: `option_letter`
8. Annotation schema: `bbox`
9. Program schema: `select_option(cyclic_order.equivalent_loop, rule=rotation_allowed_reflection_disallowed, options=4); scene=cyclic_order; scope=cyclic_order_equivalent_label`

## Program Contract

Program: `select_option(cyclic_order.equivalent_loop, rule=rotation_allowed_reflection_disallowed, options=4); scene=cyclic_order; scope=cyclic_order_equivalent_label`

Candidate set: the visible cyclic-order tokens, reference loop, gap/swap markers, numbered positions, and labeled options inside the `cyclic_order_equivalent_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `cyclic_order`, `equivalent_loop`, `rotation_allowed_reflection_disallowed`, `cyclic_order_equivalent_label`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the unique valid option label.
Annotation witnesses: `annotation` uses the `bbox` schema; one image-pixel bounding box around the matching option-loop image.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `transformation`, `matching`

## Query Contract

- Supported public `query_id`: `single`
- Query variation is not semantic for this task. Token rendering style, loop path style, scene variant, answer label position, bead count, and palette/theme choices are generation or render axes recorded in trace metadata.

## Generation Contract

- The renderer shows one reference loop and exactly four labeled option loops.
- Exactly one option is equivalent to the reference up to cyclic rotation.
- Every distractor breaks cyclic order by construction.
- Token count is sampled from `4..5`.
- Color-bearing token styles maintain minimum Lab color distance `>= 50`.
- Supported token render styles are `colored_beads`, `shape_tokens`, `colored_shape_tokens`, `outline_shape_tokens`, and `symbol_badges`.
- Supported loop path styles are `ellipse`, `rounded_rect`, `polygon_loop`, `wavy_loop`, and `beaded_string`.
- Supported scene variants are `necklace_board`, `charm_card_grid`, `route_loop_diagram`, and `token_ring_outline`.

## Prompt Contract

- Bundle: `puzzles_cyclic_order_v1`
- `scene_key`: `cyclic_order`
- `task_key`: `cyclic_order_equivalent_label_query`
- `query_key`: `single`
- Prompt-facing answer is the unique valid option label.
- Prompt-facing annotation is one image-pixel bounding box around the matching option-loop image.

## Annotation + Trace Contract

- `answer_gt.type`: `option_letter`
- `annotation_gt.type`: `bbox`
- `projected_annotation` includes `bbox` and `pixel_bbox`.
- `render_map.option_choice_bboxes_px` stores option image bboxes keyed by `option_choice_id`.
- `execution_trace` records the public query, token/render axes, option specs, answer option id/label, valid option id, and solver trace.
- Answer and annotation are both projected from the same valid option id.

## Determinism

- Deterministic sampling/rendering from `instance_seed`, scene config, prompt bundle, and code version.
- No semantic auto-relaxation is used to force acceptance.
