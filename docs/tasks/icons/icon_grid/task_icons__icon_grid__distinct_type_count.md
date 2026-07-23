# `task_icons__icon_grid__distinct_type_count`

## Identity
- domain: `icons`
- scene_id: `icon_grid`
- module: `src/trace_tasks/tasks/icons/icon_grid/distinct_type_count.py`
- prompt bundle: `icons_icon_grid_v1`

## Program Contract

Program: `count.distinct_categories(scene=icon_grid, scope=visible_grid_cells, category=icon_type, output=count)`

Candidate set: occupied cells in the visible icon grid.
Operands: visible icon type identity for every occupied grid cell. All icons in one generated instance share the same rendered color, so color is not a counting cue.
Operation: count how many distinct icon types appear in occupied grid cells; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema.
Annotation schema: `bbox_set`.
Annotation witnesses: `annotation` uses one whole grid-cell bbox per distinct icon type. For each type, choose the representative occupied cell in the topmost row; if tied, choose the leftmost such cell.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Notes
The scene uses visible grid cells so representative-category annotation targets
are stable rectangular cells, not tight icon-object boxes.
The rendered color is held constant within each image to isolate icon-type counting.
