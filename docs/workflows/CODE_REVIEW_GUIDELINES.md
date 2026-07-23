# Code Review Guidelines

- Keep task behavior deterministic and versioned.
- Do not hardcode user-facing prompts in task modules.
- Keep answers, annotations, witnesses, and verifier payloads consistent.
- Reject hidden or unrecorded randomness.
- Prefer existing shared helpers over duplicated utilities.
- Place new helpers at the narrowest reusable ownership boundary.
- Keep task-specific rendering and semantics inside the owning scene package.
- Add regression coverage for behavior and contract changes.
- Update source-of-truth documentation in the same change.
- Do not commit generated datasets, credentials, checkpoints, or local paths.
