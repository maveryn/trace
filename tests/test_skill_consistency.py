from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_skill_consistency.py"
SPEC = importlib.util.spec_from_file_location("check_skill_consistency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def _write_skill(root: Path, *, skill_md: str, openai_yaml: str) -> Path:
    skill_dir = root / ".agents" / "skills" / "trace-example"
    (skill_dir / "agents").mkdir(parents=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "EXAMPLE.md").write_text("# Example\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    (skill_dir / "agents" / "openai.yaml").write_text(openai_yaml, encoding="utf-8")
    return skill_dir


VALID_SKILL = """---
name: trace-example
description: Demonstrate a valid Trace workflow. Use when testing the public skill checker.
---

# Example

Read `docs/EXAMPLE.md`, then use `$trace-example` for the handoff.
"""

VALID_INTERFACE = """interface:
  display_name: "Trace Example"
  short_description: "Exercise the public skill checker"
  default_prompt: "Use $trace-example to validate this example workflow."
"""


def test_public_skills_are_consistent() -> None:
    assert CHECKER.validate_repository(REPO_ROOT) == []


def test_valid_skill_and_interface_pass(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        skill_md=VALID_SKILL,
        openai_yaml=VALID_INTERFACE,
    )
    assert (
        CHECKER.validate_skill(
            skill_dir,
            repo_root=tmp_path,
            known_skills={"trace-example"},
        )
        == []
    )


@pytest.mark.parametrize(
    ("skill_md", "expected"),
    [
        (
            VALID_SKILL.replace("description:", "metadata: extra\ndescription:"),
            "frontmatter must contain exactly",
        ),
        (
            VALID_SKILL.replace(
                "docs/EXAMPLE.md",
                "/" + "home" + "/example/trace/docs/EXAMPLE.md",
            ),
            "machine-specific absolute path",
        ),
        (
            VALID_SKILL.replace("docs/EXAMPLE.md", "C:\\work\\trace\\docs\\EXAMPLE.md"),
            "machine-specific Windows path",
        ),
        (
            VALID_SKILL.replace("$trace-example", "$trace-missing"),
            "unknown cross-skill reference",
        ),
        (
            VALID_SKILL.replace("docs/EXAMPLE.md", "docs/MISSING.md"),
            "referenced document does not exist",
        ),
    ],
)
def test_skill_content_drift_is_rejected(
    tmp_path: Path,
    skill_md: str,
    expected: str,
) -> None:
    skill_dir = _write_skill(
        tmp_path,
        skill_md=skill_md,
        openai_yaml=VALID_INTERFACE,
    )
    errors = CHECKER.validate_skill(
        skill_dir,
        repo_root=tmp_path,
        known_skills={"trace-example"},
    )
    assert any(expected in error for error in errors)


@pytest.mark.parametrize(
    ("openai_yaml", "expected"),
    [
        (
            VALID_INTERFACE.replace(
                '  display_name: "Trace Example"\n',
                '  icon_small: "./assets/icon.png"\n'
                '  display_name: "Trace Example"\n',
            ),
            "must contain exactly",
        ),
        (
            VALID_INTERFACE.replace('"Exercise the public skill checker"', "short"),
            "must be double-quoted",
        ),
        (
            VALID_INTERFACE.replace("$trace-example", "$different-skill"),
            "must mention $trace-example",
        ),
    ],
)
def test_interface_drift_is_rejected(
    tmp_path: Path,
    openai_yaml: str,
    expected: str,
) -> None:
    skill_dir = _write_skill(
        tmp_path,
        skill_md=VALID_SKILL,
        openai_yaml=openai_yaml,
    )
    errors = CHECKER.validate_skill(
        skill_dir,
        repo_root=tmp_path,
        known_skills={"trace-example"},
    )
    assert any(expected in error for error in errors)
