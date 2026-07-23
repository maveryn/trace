# `task_puzzles__cyclic_order__insertion_position_label`

## Taxonomy

1. Domain: `puzzles`
2. Scene package: `src/trace_tasks/tasks/puzzles/cyclic_order/`
3. Scene id: `cyclic_order`
4. Public task id: `task_puzzles__cyclic_order__insertion_position_label`
5. Objective contract: `insertion_position_label`
6. Query ids: `single`
7. Answer schema: `option_letter`
8. Annotation schema: `bbox`
9. Program schema: `select_option(cyclic_order.insertion_position, reference=reference_loop, partial=labeled_gap_loop_missing_one_token, action=insert_missing_reference_token); scene=cyclic_order; scope=insertion_position_label`

## Program Contract

Program: `select_option(cyclic_order.insertion_position, reference=reference_loop, partial=labeled_gap_loop_missing_one_token, action=insert_missing_reference_token); scene=cyclic_order; scope=insertion_position_label`

Candidate set: the visible cyclic-order tokens, reference loop, gap/swap markers, numbered positions, and labeled options inside the `insertion_position_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `cyclic_order`, `insertion_position`, `reference`, `reference_loop`, `partial`, `labeled_gap_loop_missing_one_token`, `action`, `insert_missing_reference_token`, `insertion_position_label`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `state_update`, `matching`

## Query Contract

- `single`: choose the single gap option that repairs the partial cyclic order.

Token rendering style, loop path style, scene variant, answer option position, and theme are generation/render axes recorded in trace metadata. They are not query ids.

## Answer Contract

- Type: `option_letter`
- Value: one of `A`, `B`, `C`, or `D`.
- There is exactly one valid option by construction.

## Annotation Contract

- Type: `bbox`
- Value: image-pixel bounding box `[x0, y0, x1, y1]` around the selected labeled gap badge.
- The annotation marks the selected in-scene gap badge.

## Prompt Contract

- Bundle: `puzzles_cyclic_order_v1`
- `scene_key`: `cyclic_order`
- `task_key`: `insertion_position_label_query`
- Query key: `insertion_position`

## Generation Contract

- The reference loop has five unique tokens.
- The partial loop shows the other four tokens in cyclic order with four labeled gaps.
- The missing token is the one present in the reference loop but absent from the partial loop.
- Gap badges `A-D` are drawn directly on the partial loop.
- Candidate insertion sequences are validated against the reference cyclic order up to rotation.
