---
name: trace-task-unit-audit
description: Audit Trace task boundaries, query ids, program schemas, answer schemas, and annotation schemas. Use when deciding whether a task or branch should be kept, split, merged, renamed, or retired under the public taxonomy.
---

# Trace Task-Unit Audit

## Read first

1. `docs/contracts/TAXONOMY.md`
2. `docs/contracts/TASK_UNIT_POLICY.md`
3. `docs/contracts/PROGRAM_SCHEMA_CATALOG.md`
4. `docs/contracts/SOURCE_LAYOUT.md`
5. The applicable document under `docs/domains/`

## Audit

1. Identify the stable `domain`, `scene_id`, `task_id`, program, prompt scaffold, answer schema, and annotation schema.
2. Keep a `query_id` internal only when it is a narrow semantic branch of the same program, prompt meaning, schemas, and rendered scene grammar.
3. Split a branch that changes the core program, output schema, witness roles, prompt scaffold, candidate type, or visible grammar.
4. Merge tasks only when those same contract axes agree; do not merge merely because prose or implementation helpers overlap.
5. Use concrete existing program-schema vocabulary before proposing a new schema.
6. Audit without implementing unless the user explicitly requested both.

Report `Decision`, `Contract reason`, `Program schema`, `Annotation fit`, and `Docs/code follow-up` for each audited unit. Update domain policy in its source document rather than this skill.
