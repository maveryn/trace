# `task_icons__icon_grid__distinct_color_count`

## Identity
- domain: `icons`
- scene_id: `icon_grid`
- module: `src/trace_tasks/tasks/icons/icon_grid/distinct_color_count.py`
- prompt bundle: `icons_icon_grid_v1`

## Program Contract

Program: `count.distinct_categories(scene=icon_grid, scope=visible_grid_cells, category=rendered_color, output=count)`

Candidate set: occupied cells in the visible icon grid.
Operands: rendered icon color for every occupied grid cell. All icons in one generated instance share the same icon type, so shape is not a counting cue.
Operation: count how many distinct rendered icon colors appear in occupied grid cells; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema.
Annotation schema: `bbox_set`.
Annotation witnesses: `annotation` uses one whole grid-cell bbox per distinct icon color. For each color, choose the representative occupied cell in the topmost row; if tied, choose the leftmost such cell.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Notes
The scene uses visible grid cells so representative-category annotation targets
are stable rectangular cells, not tight icon-object boxes.
The icon type is held constant within each image to isolate rendered-color counting.
