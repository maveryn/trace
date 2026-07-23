# Illustrations Domain Contract

Use this document for illustration-domain rules. Exact active scenes and tasks
live in `docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/illustrations/`.

## Scope
Illustrations covers synthetic drawings of recognizable objects, environments,
object parts, and derived visual-composition tasks built from those scenes. It
does not cover natural images, pure icon grids, or arbitrary caption/OCR tasks.

Use `illustrations` when semantic object/part records and their rendered pixel
geometry are the source of truth. Use `icons` for abstract reusable icon fields
and `pages` for document-like layouts.

## Scene Boundary
A scene is the stable illustration grammar: object catalog, environment type,
part vocabulary, reference/cutout/option scaffold, or scene-comparison layout.
Visual styles such as vector, top-down pixel, isometric pixel, palette, and
background treatment may vary inside a scene when the same verifier records are
available.

Derived composition scenes, such as cutout reconstruction or missing patch
options, are valid scenes when the public task is about the composition
interface rather than the source illustration category. Record the source scene
or renderer as metadata, not as a public task identity layer.

## Task And Query Boundary
Split tasks when the objective changes between object count, part count,
visible-part reasoning, spatial relation, scene difference, missing patch,
cutout matching, or option selection. Object category or style variation may
remain a parameter when the answer/annotation/program contract is stable.

Avoid broad tasks that collapse rich illustrations into icon-like collections.
If a task can be represented equally well as an icon-field count, either move it
to `icons` or make the illustration scene provide richer object/context
grounding.

## Annotation Policy
Prompt-facing annotation should mark visible object bboxes, semantic part
bboxes, paths, regions, source/candidate panels, or other decisive witnesses in
final-image coordinates. Use map annotation when source and target,
before/after, reference/candidate, or object/part roles matter.

Do not annotate decorative sky/sun/cloud elements unless a task explicitly
promotes them to foreground semantic objects.

## Rendering
Objects and parts must be drawn from the same trace records used by verifiers.
Environment scenes should keep placements natural, such as sky-capable objects
in sky bands, vehicles on roads, water objects in water, and land objects on
valid ground/surface regions.

Illustration scenes should support the shared render-only canvas profiles by
default: landscape `1200x800`, square `960x960`, and portrait `800x1200`.
Canvas profile is a render/style axis, not a query id or public task split.
Derived reconstruction tasks should render their source scene directly at the
selected profile and then downscale the final option layout only if needed to
stay under the 1,280,000-pixel rendering cap, scaling annotation coordinates with the image.
Missing-patch source crops should be sized relative to the resolved source image:
width `15%-30%`, height `15%-26%`, with area capped at `6.5%`.
For quarter-turn rotated-tile tasks, choose profile-aware square-cell grids:
landscape `2x3`, square `3x3`, and portrait `3x2`. Top-down RPG tile
scenes use shared 48px tile profiles: landscape `27x18`, square `21x21`,
and portrait `18x27`.

The rendered source illustration itself should not be placed on a decorative
outer background, card, border, or frame. If a visual-option task needs layout
space for choices, keep that wrapper functional and minimal: option labels,
missing-region masks, tile grid lines, and tight gutters are acceptable, but
source-panel titles, decorative outlines, and worksheet-like card backgrounds
should not be added to the scene image.

Style and background variation must remain non-semantic unless queried. Dense
scenes should cap foreground object counts to preserve readable object and part
bboxes.

Isometric illustration scenes should use the shared 25-tone neutral background
pool through the illustrations isometric visual-style adapter. The tone may
control canvas, terrain edge/shadow, and label colors, but semantic terrain
colors such as grass, water, dock, rock, crop, and soil must remain recognizable.
Record the actual `background_tone_id` and RGB role metadata in the trace.

## Shared Code
Reusable object catalogs, scene grammars, environment layout, part metadata, and
renderer/style helpers belong under `src/trace_tasks/tasks/illustrations/shared/`.
Scene-local shared modules should own scaffolds tied to one illustration scene.
