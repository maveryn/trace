# `task_three_d__surface_fixture__repeated_element_count`

## Summary
- Domain: `three_d`
- Scene id: `surface_fixture`
- Scene package: `surface_fixture`
- Query id: `single`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(surface_fixture_elements, present=true, element_type=target_element_type)); scene=surface_fixture; scope=repeated_element_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `repeated_element_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `surface_fixture_elements`, `present`, `true`, `element_type`, `target_element_type`, `surface_fixture`, `repeated_element_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Contract
The image shows one projected fixture surface with one visible element family:
tiles, holes, slots, compartments, vents, windows, doors, drawer pulls, bricks,
pavers, lockers, mailboxes, drive bays, buttons, solar panels, screws, hex
nuts, washers, sockets, hooks, indicator lights, brackets, U-bolts, or pipes.
The prompt asks directly for the number of visible elements of that family.

The scene variant determines the counted element type:

- `wall_tile_panel` counts `tile`
- `perforated_panel` counts `hole`
- `slot_board` counts `slot`
- `compartment_tray` counts `compartment`
- `vent_panel` counts `vent`
- `window_grid` counts `window`
- `door_bank` counts `door`
- `drawer_pull_panel` counts `drawer_pull`
- `brick_wall` counts `brick`
- `paver_floor` counts `paver`
- `locker_bank` counts `locker`
- `mailbox_bank` counts `mailbox`
- `server_rack` counts `drive_bay`
- `control_panel` counts `button`
- `solar_panel_array` counts `solar_panel`
- `screw_plate` counts `screw`
- `hex_nut_plate` counts `hex_nut`
- `washer_plate` counts `washer`
- `socket_bank` counts `socket`
- `hook_board` counts `hook`
- `indicator_light_panel` counts `light`
- `bracket_panel` counts `bracket`
- `u_bolt_plate` counts `u_bolt`
- `pipe_rack` counts `pipe`

The answer is the integer count of finalized elements whose `element_type`
matches the sampled `target_element_type`. Pixels are render output, not
verifier source of truth.

Visible elements are assigned canonical named colors for visual variety. For
this task, color is recorded as `non_semantic_visual_variation`; it is not part
of the count predicate and should not be used to filter the counted set.

The placement family is sampled by scene variant and recorded as
`layout_family`; the concrete placement style is recorded as `layout_style`.
The shared scene layout families are:

- `strict_grid`: manufactured grid surfaces such as windows, lockers, server
  racks, solar panels, compartments, doors, and mailboxes. Styles:
  `uniform_grid`, `variable_grid`.
- `tiled_staggered`: surface patterns such as bricks, pavers, wall tiles, and
  perforated panels. Styles: `uniform_grid`, `variable_grid`, `brick_grid`.
- `loose_mounted_rows`: rails or boards with mounted hardware such as slots,
  vents, drawer pulls, sockets, hooks, brackets, U-bolts, pipes, and some
  control/light panels. Styles: `jittered_grid`, `loose_rows`, `uniform_grid`.
- `panel_scatter`: individually mounted pieces such as screws, washers,
  hex nuts, control buttons, and some indicator lights. Styles:
  `panel_scatter`, `jittered_grid`.

One scene variant may allow multiple layout families when both are plausible.
For example, `indicator_light_panel` can be loose rows or scatter, while
`perforated_panel` is restricted to `tiled_staggered` and does not use random
scatter.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around
each counted surface element. The fixture panel, screw heads, and background
context are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_surface_fixture_v1` under `src/trace_tasks/resources/prompts/three_d/surface_fixture/`.
The trace records scene variant, target element type, target element ids,
surface projection metadata, layout family, layout style, visual color names
and counts, projected element boxes and centers, and the solver count
predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config
defaults, prompt bundle, and code versions. Answers and annotation come from the
same finalized fixture trace.
