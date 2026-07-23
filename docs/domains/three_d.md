# Three-D Domain Contract

Use this document for three_d-domain rules. Exact active scenes and tasks live
in `docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/three_d/`.

## Scope
`three_d` covers rendered 3D scenes with explicit camera pose, world-space
geometry, projection metadata, and metadata-grounded verifiers. Tasks reason
over spatial relations, camera distance, height, occlusion, support surfaces,
multi-view correspondence, object counts, fixtures, conveyor scenes,
streets, rooms, or warehouse layouts.

Use `three_d` for perspective 3D environments. Use `geometry` for abstract
geometric solids and `puzzles` for abstract 3D puzzle boards.

## Scene Boundary
A three_d scene is the stable 3D environment grammar: object scene, object
cluster, surface fixture, conveyor, room, street, warehouse, or
another camera-projected world. Scene variants may vary room/platform type,
camera orbit band, surface style, object profiles, lighting, or palette when the
same projection and verifier contract holds.

Create a new scene when the environment grammar, camera/view interface,
candidate option surface, or world-coordinate verifier changes materially.

## Task And Query Boundary
Split tasks when the program changes between object count, attribute count,
distance/rank selection, relation, occlusion order, height extremum, multi-view
matching, fixture count, route/lane relation, or warehouse path reasoning.

Valid `query_id` axes include closest/farthest, left/right, in-front/behind,
same/different support, arithmetic operator over the same operand roles, or
bounded relation direction inside one stable program. Literal target values
inside one visual channel, such as one object type or one semantic color, are
sampled operands. Switching visual reasoning channels or predicate arity, such
as object-type match vs color match or single-attribute match vs color+type
binding, is a public task split when it changes how the model must scan the
image. Object profile, camera pose, room style, and surface style are
metadata/style axes unless directly queried.

## Annotation Policy
Prompt-facing annotation should mark projected visible objects, fixtures,
reference surfaces, candidate markers, option panels, or route/path witnesses in
final-image coordinates. Use map annotation when reference/candidate,
source/target, before/after, left/right view, or operand roles matter.

Answers must come from finalized 3D metadata such as world coordinates, camera
distances, support assignment, projected bboxes, or occlusion ordering, not from
pixel inference.

## Rendering And Assets
Object profiles should be visually recognizable enough for prompt names and
should expose stable projected bboxes or point markers. Semantic color, size,
height, depth, support, and relation predicates must be recorded in trace
metadata when queried.

Three_d scenes should render from one of the canonical source-canvas presets by
default: `1200x800` landscape, `800x1200` portrait, or `960x960` square. The
sampled preset is render metadata, not a query axis or task split. If a task
composes multiple panels/options, each source panel should use the same sampled
canonical preset unless the task has a documented reason to override it; the
final composed image may expand or downscale to stay under the `1,280,000`
pixel domain cap while projecting all answer and annotation coordinates after
final placement.

Prompt-facing named colors must use the repo-wide canonical 10-color palette
from `trace_tasks.tasks.shared.named_colors` and should be rendered in prompts as
`<color name> [#RRGGBB]`. Scene-specific palettes are acceptable only for
unnamed visual/style variation that is not sampled as a named answer or prompt
predicate.

Named object count/readout tasks should use shared visual-confusion policies
from `src/trace_tasks/tasks/three_d/shared/` when sampling wrong-type distractors. If the
target object belongs to a near-confusable family, such as card/envelope/book,
sphere/button, cup/bowl/tray, lantern/candle, or pencil/ruler, same-family
objects should not be sampled as semantic wrong-type distractors for that
target. Scene-specific object pools may be narrower for layout reasons, but
confusion filtering should remain shared.

Use `THREE_D_NAMED_OBJECT_SHAPE_TYPES` from
`src/trace_tasks/tasks/three_d/shared/object_resources.py` as the canonical curated named
object pool for prompt-facing named-object count/readout tasks. Scene-specific
compatibility pools, such as `OBJECT_SCENE_NAMED_CANDIDATE_SHAPE_TYPES`, should
only be derived intersections with renderer/layout support. Broad renderable
small-object pools are not named-question pools.

Inspect representative generated samples for object fidelity. Style, lighting,
camera, and object-palette variation must not encode answer value, query id,
correct option, relation truth, or construction order unless explicitly queried.

Neutral floor/canvas tone variation is domain-shared. All active three_d scene
configs inherit the approved 25-tone pool from `src/trace_tasks/resources/configs/domains/three_d/base.yaml`:
20 light matte/studio/industrial tones and 5 dark graphite/warehouse tones. Use
`trace_tasks.tasks.three_d.shared.visual_styles.resolve_three_d_surface_tone` through
the scene render-parameter resolver instead of hardcoding scene-local floor,
grid, edge, text, or canvas colors. Scene-local configs should not pin
`floor_rgb`, `grid_rgb`, `edge_rgb`, `text_rgb`, or `text_stroke_rgb`; street
sidewalk/curb and warehouse aisle/shelf-zone surfaces should also inherit from
the selected tone. Semantic non-surface colors such as road asphalt, route path
highlighting, and conveyor belt style may remain scene-specific. Each tone owns
readable `text_rgb` and `text_stroke_rgb` values so dark treatments do not
inherit light-scene label defaults. Conveyor-like belt scenes should use the
shared named conveyor belt styles from the same module; belt style is render
metadata, not a task/query axis.

Object-cluster instances may apply a bounded per-object `orientation_deg`
rendering jitter for visual variety. This value must be recorded in trace
metadata and projected-geometry calculations, but remains a renderer axis rather
than a query/task split unless a task explicitly asks about object orientation.

## Shared Code
Reusable camera, projection, object-profile, room, street, warehouse, and
fixture helpers belong under `src/trace_tasks/tasks/three_d/shared/` when reused across
scenes. Scene-local shared modules should own environment-specific construction,
rendering, and annotation projection.
