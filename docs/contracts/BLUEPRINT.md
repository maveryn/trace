# Trace Blueprint

Status: normative contract for dataset ABI, determinism, and build policy.

## 1) Goal
Trace generates grounded visual-reasoning instances with:

1. prompt,
2. typed answer,
3. image artifacts,
4. typed annotation,
5. sidecar trace for replay and verifier metadata.

## 2) Public Taxonomy
Public dataset taxonomy uses:

```text
domain -> scene_id -> task_id
```

Rules:

1. `scene_id` is the visual rendering grammar.
2. `task_id` is the public sampling unit and uses
   `task_<domain>__<scene_id>__<objective_contract>`.
3. `query_id` is replay and branch-diagnostic metadata inside one task
   contract, not a public sampling unit.
4. Source layout, config files, prompt bundle paths, registered task classes,
   and registered source scene/group fields are implementation routing, not
   public taxonomy.
5. Current source layout uses:
   - `src/trace_tasks/tasks/<domain>/<scene_id>/<objective_contract>.py`
   - `src/trace_tasks/tasks/<domain>/<scene_id>/shared/`
   - `src/trace_tasks/resources/configs/domains/<domain>/<scene_id>.yaml`
   - `src/trace_tasks/resources/prompts/<domain>/<scene_id>/<bundle_id>.json`

## 3) Required Artifacts
### 3.1 Train Instance
Every record must include:

1. identity and taxonomy (`instance_id`, `instance_seed`, `domain`,
   `scene_id`, `task`/`task_id`, optional `query_id`);
2. `prompt` and optional `prompt_variants`;
3. `images[]` with dataset-root-relative `path` and `image_hash`;
4. `answer_gt: {type, value}`;
5. `annotation_gt: {type, value}`;
6. `reward_contract`;
7. `trace_ref`;
8. `versions`.

Constraints:

1. `answer_gt.type` and `annotation_gt.type` must be registered type ids.
2. `images[*].path` is dataset-root-relative, never absolute.
3. `trace_ref` is mandatory.
4. `reward_contract` is mandatory and must match the resolved public
   answer/annotation reward mapping.

### 3.2 Sidecar Trace
Trace payload is mandatory and referenced by `trace_ref`.

Required sections:

1. `scene_ir`
2. `query_spec`
3. `render_spec`
4. `render_map`
5. `execution_trace`
6. `witness_symbolic`
7. `projected_annotation`

## 4) Prompt Contract
1. Prompt text must live in external bundles under `src/trace_tasks/resources/prompts/`.
2. Deterministic composition layers are scene, task, optional query, and output
   mode (`answer_only`, `answer_and_annotation`).
3. Store rendered prompt modes in `prompt_variants` when generated.
4. Record prompt metadata in trace: bundle id, layer keys, selected variants,
   slot values, and output mode.
5. Template cardinality rule: each required list has exactly 5 high-quality
   variants unless the prompt schema explicitly declares an approved exception.

## 5) Annotation And Reward Contracts
1. Symbolic witness metadata is the source of truth in trace.
2. Prompt-facing annotation is projected from the witness after final layout and
   rendering placement are known.
3. `TrainInstance.annotation_gt` uses one registered public annotation type.
4. Annotation-order semantics are task-defined and deterministic.
5. Unsupported annotation-type requests are hard validation errors.
6. `reward_contract` is builder-owned metadata derived from `answer_gt.type`
   and `annotation_gt.type`.
7. Reward-contract ids are versioned separately from answer/annotation types.

## 6) Determinism And Identity
1. Single root seed per instance: `instance_seed`.
2. All sub-seeds derive by namespace hash.
3. No hidden randomness; all randomness is explicit and recorded when it affects
   replay, layout, rendering, prompt selection, answer, or annotation.
4. Canonical JSON (RFC 8785 JCS) is required for identity hashes.
5. Hash algorithm is `blake3`.
6. `instance_id` uses semantic fields and image content hashes, not file paths.

## 7) Visual Variation Invariant
1. Repeated-unit visual grammars should include non-semantic unit-size jitter
   whenever the image is built from repeated cells, tiles, slots, grid squares,
   hexes, stickers, or voxels.
2. The first target is a sampled minimum-to-maximum rendered unit-size span of
   at least `2x`; narrower ranges are acceptable only when readability, fit, or
   annotation integrity requires them.
3. Unit-size jitter must be explicit and recorded in render metadata.
4. Annotation bboxes/points must be projected after final jittered layout.

## 8) Sampling Policy
1. Global sampling unit is `task_id`.
2. Domain and scene probabilities are derived by task aggregation unless build
   config explicitly overrides them.
3. Query sampling is inside each task and uniform by default.
4. Uniform semantic sampling is implemented by seeded RNG draws over explicit
   equal probabilities. Seed modulo, hash modulo, cursor cycling, and index
   enumeration are not valid substitutes for random sampling of task-internal
   semantic axes.
5. Exact stratification, when needed for a build, belongs in the build sampler
   and must be recorded as a sampling policy rather than hidden inside task
   generation.
6. Validate answer distributions per task/query with lightweight
   anti-degeneracy checks over generated answers.
7. Never relax semantic constraints to force acceptance.

## 9) Build, Validation, And Finalize
1. Build to staging directory.
2. Generate train records and trace shards.
3. Run pre-finalize validation.
4. Optionally run strict reproducibility pass.
5. Atomically finalize on success.
6. Persist failure bundle on failure.

Required build outputs:

1. `validation_report.json`
2. `build_report.json`

## 10) Versioning Policy
1. `instance_version` is the ABI contract version.
2. Breaking schema or semantic changes require a version bump.
3. Mixed instance versions in one dataset are invalid.
4. Replay-critical versions must be recorded in `versions`.

## 11) Quality Gates
A task/build is acceptable only if:

1. schema and trace linkage are valid,
2. answer/annotation/witness come from one execution trace,
3. final answer is unique by construction,
4. determinism checks pass for fixed seeds,
5. prompt metadata and template constraints pass,
6. no silent constraint relaxation is used.

## 12) Document Ownership
When contracts change, update the matching source-of-truth docs:

1. `docs/contracts/SYSTEM_ARCHITECTURE.md` for implementation boundaries;
2. `docs/workflows/TASK_AUTHORING.md` for author workflow;
3. `docs/workflows/BUILD_VALIDATION.md` for operational validation;
4. domain/task docs for domain-specific behavior.
