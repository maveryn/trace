# Contract Docs

This folder contains repo-wide contracts that should change only when Trace
runtime semantics change. Keep workflow instructions, transition plans, generated
inventories, and domain-specific policy outside `docs/contracts/`.

- `BLUEPRINT.md` — dataset ABI, determinism, build, and quality gates.
- `SYSTEM_ARCHITECTURE.md` — runtime layers, module boundaries, and lifecycle.
- `SOURCE_LAYOUT.md` — current task-source layout and shared-code ownership.
- `TAXONOMY.md` — public `domain -> scene_id -> task_id` taxonomy.
- `TASK_UNIT_POLICY.md` — task/query boundary and merge/split rules.
- `PROGRAM_SCHEMA_CATALOG.md` — reusable program schemas and the canonical
  task-level reasoning-operation metadata vocabulary.
- `PROMPT_SYSTEM.md` — prompt asset schema, composition, and metadata.
- `ANNOTATION_AND_REWARD_CONTRACTS.md` — annotation type selection and
  answer/annotation reward dispatch contract.
- `VALIDATION_ERROR_CODES.md` — validation error-code catalog.
