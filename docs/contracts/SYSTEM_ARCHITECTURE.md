# Trace System Architecture

Implementation map for the contracts in `docs/contracts/BLUEPRINT.md`.

## 1) Runtime Layers
1. `src/trace_tasks/core/` — deterministic infrastructure: types, hashing, seeds,
   taxonomy, validation, build, reward contracts, reward scoring, and export.
2. `src/trace_tasks/core/prompts/` — prompt bundle loading, validation, selection, and
   rendering.
3. `src/trace_tasks/core/visual/` — deterministic shared background and post-image noise.
4. `src/trace_tasks/tasks/` — task registry plus concrete task implementations.
5. `src/trace_tasks/tasks/shared/` — cross-domain task helpers.
6. `src/trace_tasks/tasks/<domain>/shared/` — domain-wide helpers only when reused across
   multiple scenes in that domain.
7. `src/trace_tasks/tasks/<domain>/<scene_id>/` — current-layout task files and
   scene-local `shared/` helpers.
8. `src/trace_tasks/resources/prompts/` — external prompt assets.
9. `src/trace_tasks/resources/configs/` — packaged generation and rendering
   defaults; `examples/configs/` contains example build configurations.
10. `src/trace_tasks/review/` — optional contributor review recipes, replay,
    audits, local application, portable calibration, and report export. It
    consumes public task outputs and verifier contracts; it does not define
    task semantics or paper evaluation policy.

All active task work uses the current source layout described in
`docs/contracts/SOURCE_LAYOUT.md`.

## 2) Runtime Data Flow
1. Load build config and type registry.
2. Resolve deterministic `dataset_id`.
3. Sample a public `task_id`.
4. Generate staged instances:
   - explicit task params from build config,
   - task-local query sampling,
   - public taxonomy resolution (`domain -> scene_id -> task_id`),
   - scene/task/query prompt rendering,
   - image rendering and visual variation,
   - answer and annotation binding from the same execution trace,
   - reward-contract resolution from public answer/annotation types,
   - sidecar trace write,
   - train-record write with `trace_ref`.
5. Optionally run strict-repro second pass and compare.
6. Run pre-finalize validation.
7. Write `validation_report.json` and `build_report.json`.
8. Atomically finalize on success; persist failure bundle on error.

## 3) Core Module Responsibilities
### Core Runtime
1. `src/trace_tasks/core/types.py` — ABI dataclasses.
2. `src/trace_tasks/core/canonical.py` and `src/trace_tasks/core/hash_utils.py` — canonical
   hashing.
3. `src/trace_tasks/core/seed.py` — seed derivation and spawn helpers.
4. `src/trace_tasks/core/identity.py` — `instance_id` computation.
5. `src/trace_tasks/core/type_registry.py` — answer/annotation type checks.
6. `src/trace_tasks/core/trace_store.py` — sidecar trace shard I/O.
7. `src/trace_tasks/core/validation.py` — pre-finalize dataset validation.
8. `src/trace_tasks/core/builder.py` — build orchestration.
9. `src/trace_tasks/core/build_presets.py` — reusable build recipes.
10. `src/trace_tasks/core/reward_contracts.py` — public reward-contract resolver.
11. `src/trace_tasks/core/reward_scoring.py` — shared Trace answer/annotation scoring.
12. `src/trace_tasks/core/rlvr_export.py` — Trace-to-RLVR export helpers.
13. `src/trace_tasks/core/taxonomy.py` — public taxonomy and implementation/source
    routing metadata.
14. `src/trace_tasks/core/strict_repro.py` — strict reproducibility comparisons.
15. `src/trace_tasks/core/scene_config.py` — config resolver for scene defaults.
16. `src/trace_tasks/core/sampling.py` — shared sampling primitives.
17. `src/trace_tasks/core/json_io.py` — deterministic JSON writing.

### Prompt And Visual
1. `src/trace_tasks/core/prompts/assets.py` — prompt bundle loading/cache.
2. `src/trace_tasks/core/prompts/schema.py` — prompt schema validation.
3. `src/trace_tasks/core/prompts/select.py` — deterministic variant selection.
4. `src/trace_tasks/core/prompts/render.py` — strict rendering and metadata.
5. `src/trace_tasks/core/visual/background.py` — background style selection/rendering.
6. `src/trace_tasks/core/visual/noise.py` — post-image noise selection/application.
7. `src/trace_tasks/core/visual/defaults.py` — visual defaults loading.

### Task Framework
1. `src/trace_tasks/tasks/registry.py` — registration and creation.
2. `src/trace_tasks/tasks/base.py` — task protocol and `TaskOutput`.
3. `src/trace_tasks/tasks/shared/*` — cross-domain query, layout, annotation, config,
   prompt, and output helpers.
4. `src/trace_tasks/tasks/<domain>/<scene_id>/<objective_contract>.py` — one public task
   file per active task.
5. `src/trace_tasks/tasks/<domain>/<scene_id>/shared/*` — scene-local reusable state,
   sampling, rendering, prompt-slot, annotation, and output helpers.
6. `src/trace_tasks/tasks/<domain>/shared/*` — helpers reused by multiple scenes in one
   domain, never a dumping ground for one scene's task routing.

### Contributor Review

1. `src/trace_tasks/review/recipe.py` and `provenance.py` — deterministic,
   source-bound recipe capture and loading.
2. `src/trace_tasks/review/materialize.py` and `audits.py` — task-atomic local
   replay, semantic verification, and focused offline audits.
3. `src/trace_tasks/review/app/` — optional local-only browsing and feedback
   state with protected, contained media access.
4. `src/trace_tasks/review/calibration.py` — optional typed-answer probes against
   caller-managed OpenAI-compatible endpoints; never an approval gate.
5. `src/trace_tasks/review/export.py` — portable exports of local review state.

## 4) Active Task Inventory
The active public task surface is generated from the live registry and
taxonomy. Do not enumerate current tasks or scenes in this architecture doc.

Use `docs/ACTIVE_TASK_INVENTORY.md` for the committed generated inventory of
`domain -> scene_id -> task_id`. Regenerate it with:

```bash
python scripts/generate_active_task_inventory.py
```

## 5) Architecture Invariants
1. Determinism from config, seeds, code versions, prompt assets, and visual
   assets.
2. `TrainInstance` stays lightweight; heavy replay metadata stays in sidecar
   trace.
3. Public records expose `domain`, `scene_id`, and task identity. Source routing
   metadata is debug/runtime metadata only.
4. Answer, annotation, and witness metadata are consistent from one execution
   trace.
5. Shared helpers are reused before adding task-local utilities, but shared code
   must stay identity-free.
6. Public task files own objective-specific construction, answer binding,
   annotation binding, prompt slots, trace payload, and final `TaskOutput`.
7. Builder parallelism changes throughput only; dataset identity and finalized
   row ordering stay invariant for fixed build-critical config.

## 6) When To Update This Doc
Update when module boundaries, build lifecycle, shared invariants, or extension
points change. Do not update this doc for ordinary task additions/removals; use
the generated inventory and domain/task docs instead.
