# Codex Skills

Trace's repository skills live under `.agents/skills/` so Codex can discover
them while working anywhere in the checkout. They route work to canonical
contracts and workflows; they are not a second copy of task or domain policy.

## Available skills

- `$trace-task-design`: freeze a task contract before coding.
- `$trace-task-unit-audit`: decide task and query boundaries.
- `$trace-task-implementation`: implement a stable task contract.
- `$trace-prompt-design`: design external prompt bundles and examples.
- `$trace-verification-review`: test changes, expand shared dependencies, and
  inspect deterministic recipe artifacts.
- `$trace-code-review`: review a patch against public contracts.

Name a skill explicitly when you want that workflow, for example:

```text
Use $trace-task-design to specify this proposed task before implementation.
```

Codex may also invoke a skill when its description clearly matches the request.
The skill body should remain concise and link to the source-of-truth docs for
detailed policy.

## Verification routing

For prompt changes, `$trace-verification-review` routes through both
`docs/review/PROMPT_REVIEW.md` and `docs/contracts/PROMPT_SYSTEM.md`. Determine
the affected recipe slice from every consumer of the changed bundle or shared
resource, not only the named task. Capture a changed semantic contract twice
from one clean committed checkout and verify that both drafts match. Changes
that invalidate the accepted recipe require the complete replacement procedure
in [`docs/review/REVIEW_RECIPE.md`](../review/REVIEW_RECIPE.md).

## Maintenance

When workflow behavior changes:

1. Update the canonical contract or workflow under `docs/`.
2. Update only the skills whose routing or ordered actions changed.
3. Keep `SKILL.md` frontmatter limited to `name` and `description`.
4. Keep `agents/openai.yaml` synchronized with the skill and free of
   unrequested icons or tool dependencies.
5. Run `python scripts/check_skill_consistency.py` and the focused skill tests.

See [documentation and skill maintenance](DOCS_AND_SKILLS_MAINTENANCE.md) for
anti-drift rules.
