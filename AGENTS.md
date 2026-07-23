# Trace Agent Guide

## Project Goal

Trace is a grounded visual reasoning task environment for verifiable
post-training. Every generated instance includes a prompt, typed answer, image,
metadata-backed verifier payload, and sidecar execution-trace reference.

## Source Of Truth

- Documentation index: `docs/README.md`
- Dependencies and packaging: `pyproject.toml`
- Contributor review: `docs/review/README.md`
- Repo-local Codex workflows: `docs/workflows/CODEX_SKILLS.md`
- Domain and scene defaults:
  `src/trace_tasks/resources/configs/domains/<domain>/base.yaml` and
  `src/trace_tasks/resources/configs/domains/<domain>/<scene_id>.yaml`

## Branches

- `main` owns the stable public generation package and contracts.
- `dev` owns public contributor tooling, local review workflows, and repo-local
  Codex skills before they are promoted to `main`.
- `rlvr` owns the reviewed paper training and canonical evaluation surface.

## Engineering Rules

- Use public taxonomy `domain -> scene_id -> task_id` consistently.
- Public task ids follow `task_<domain>__<scene_id>__<objective_contract>`.
- Keep prompts in external, versioned prompt bundles.
- Keep generation deterministic and record every random source.
- Derive answers and annotations from the same execution trace.
- Ensure each generated instance has a unique final answer by construction.
- Never relax semantic constraints to force sample acceptance.
- Use metadata contracts, not pixels, as verifier ground truth.
- Reuse shared helpers at the narrowest appropriate ownership layer.
- Update the relevant contracts and task documentation when behavior changes.
- Keep generated review artifacts, workbooks, databases, endpoint responses,
  credentials, and machine-specific paths out of Git.
- Use the matching `.agents/skills/trace-*` workflow for task design,
  implementation, prompt design, verification, task-unit audit, or code review.
