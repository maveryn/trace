# Trace Public Taxonomy

Trace public metadata uses:

```text
domain -> scene_id -> task_id
```

## 1) Public Fields
1. `domain` is the broad reporting and balancing domain.
2. `scene_id` is the visible rendering grammar: board, chart, diagram, page,
   instrument, object scene, or other stable visual scaffold.
3. `task_id` is the public sampling unit and uses taxonomy-v0 form:
   `task_<domain>__<scene_id>__<objective_contract>`.
4. `query_id` is task metadata for replay and branch diagnostics inside one
   task contract. It is not a public taxonomy level or sampling unit.

Do not maintain active domain or task lists in this file. The generated public
inventory lives in `docs/ACTIVE_TASK_INVENTORY.md`.

## 2) Scene And Task Identity
The public task is the pair:

```text
scene contract + objective contract
```

The scene contract defines the visual grammar, object vocabulary, layout family,
non-semantic style support, and any stable query-facing scaffold.

The objective contract defines the answer schema, annotation schema, and concrete
program schema. The detailed boundary rules live in
`docs/contracts/TASK_UNIT_POLICY.md`.

## 3) Implementation Routing
Public taxonomy is not source layout. Implementation routing is source/debug
metadata only.

Current source layout is:

```text
src/trace_tasks/tasks/<domain>/<scene_id>/<objective_contract>.py
src/trace_tasks/tasks/<domain>/<scene_id>/shared/
src/trace_tasks/resources/configs/domains/<domain>/<scene_id>.yaml
src/trace_tasks/resources/prompts/<domain>/<scene_id>/<bundle_id>.json
```

Source routing/debug fields, including registered class domain/scene,
`source_domain`, and `source_scene_id`, are implementation metadata only. They
are not public taxonomy nodes, config grouping layers, or task-id formats.

## 4) Trace Metadata Shape
Sidecar traces store taxonomy metadata under `trace_payload["taxonomy"]`.

Public fields:
- `domain`
- `scene_id`
- `task_id`
- optional `query_id`

Routing/debug fields live under explicit nested blocks such as
`taxonomy.registered` and `taxonomy.source`. Trace consumers must not infer
implementation paths, config files, or prompt bundles from public task ids.

## 5) Sampling Rule
Equal task-level sampling is the default. Domain and scene probabilities are
derived by aggregating active public tasks unless a build config explicitly says
otherwise.

Query sampling happens inside the selected task. Query ids are uniform by
default and do not have config-level weights.

Uniform or equal-weight task-internal sampling means a seeded RNG draw over an
explicit probability map. Do not use seed modulo, hash modulo, cursor cycling,
or deterministic index enumeration as a substitute for random sampling of
semantic axes. Exact stratification belongs in the build sampler, not inside
public task generators.

## 6) Reasoning-Operation Metadata
Each public task declares a code-authoritative, multi-label decomposition over
the 13 analysis families defined in
`docs/contracts/PROGRAM_SCHEMA_CATALOG.md`. These labels support aggregate
coverage analysis; they are not public taxonomy nodes, task-boundary
rules, or sampling units. The current 1,000-task inventory remains the stable
public surface even when several tasks share operation families or outer
program scaffolds.

## 7) Source Of Truth
The active taxonomy mapping lives in `src/trace_tasks/core/taxonomy.py`. Build,
validation, export, and documentation checks should resolve public taxonomy
through that module instead of parsing task ids by string.
