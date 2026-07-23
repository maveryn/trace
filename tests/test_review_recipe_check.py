from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_review_recipe.py"
SPEC = importlib.util.spec_from_file_location("check_review_recipe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
review_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_check)

GENERATOR_PATH = Path("src/trace_tasks/tasks/fixture.py")
CONFIG_PATH = Path("src/trace_tasks/configs/fixture.json")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _committed_fixture(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    source = repo / GENERATOR_PATH
    source.parent.mkdir(parents=True)
    source.write_text(
        '''"""Original module documentation."""

def value() -> int:
    """Original function documentation."""
    answer = 1
    return answer
''',
        encoding="utf-8",
    )
    config = repo / CONFIG_PATH
    config.parent.mkdir(parents=True)
    config.write_text('{"enabled": true}\n', encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "review-check@example.test")
    _git(repo, "config", "user.name", "Review Check")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_non_executable_generator_drift_accepts_docs_comments_and_formatting(
    tmp_path: Path,
) -> None:
    repo, revision = _committed_fixture(tmp_path)
    (repo / GENERATOR_PATH).write_text(
        '''"""Revised module documentation."""

# A source comment is not part of task execution.
def value( ) -> int:
    """Revised function documentation."""
    answer=1
    return answer
''',
        encoding="utf-8",
    )

    assert review_check._non_executable_generator_drift_paths(
        repo,
        producer_revision=revision,
    ) == (GENERATOR_PATH.as_posix(),)


def test_non_executable_generator_drift_rejects_runtime_python_changes(
    tmp_path: Path,
) -> None:
    repo, revision = _committed_fixture(tmp_path)
    (repo / GENERATOR_PATH).write_text(
        '''"""Original module documentation."""

def value() -> int:
    """Original function documentation."""
    answer = 2
    return answer
''',
        encoding="utf-8",
    )

    assert (
        review_check._non_executable_generator_drift_paths(
            repo,
            producer_revision=revision,
        )
        is None
    )


def test_non_executable_generator_drift_rejects_config_changes(
    tmp_path: Path,
) -> None:
    repo, revision = _committed_fixture(tmp_path)
    (repo / CONFIG_PATH).write_text('{"enabled": false}\n', encoding="utf-8")

    assert (
        review_check._non_executable_generator_drift_paths(
            repo,
            producer_revision=revision,
        )
        is None
    )


def test_non_executable_generator_drift_rejects_file_set_changes(
    tmp_path: Path,
) -> None:
    repo, revision = _committed_fixture(tmp_path)
    added = repo / "src/trace_tasks/tasks/added.py"
    added.write_text("VALUE = 1\n", encoding="utf-8")

    assert (
        review_check._non_executable_generator_drift_paths(
            repo,
            producer_revision=revision,
        )
        is None
    )


def test_python_projection_ignores_nested_docstrings_not_runtime_strings() -> None:
    original = b'''"""Module docs."""
class Example:
    """Class docs."""
    async def value(self):
        """Function docs."""
        return "kept"
'''
    revised_docs = b'''"""Revised module docs."""
class Example:
    """Revised class docs."""
    async def value(self):
        """Revised function docs."""
        return "kept"
'''
    revised_runtime_string = revised_docs.replace(b'return "kept"', b'return "changed"')

    original_ast = review_check._python_executable_ast(original, filename="fixture.py")
    assert original_ast == review_check._python_executable_ast(
        revised_docs, filename="fixture.py"
    )
    assert original_ast != review_check._python_executable_ast(
        revised_runtime_string, filename="fixture.py"
    )
    assert (
        review_check._python_executable_ast(b"def invalid(:\n", filename="fixture.py")
        is None
    )


def test_non_executable_generator_drift_rejects_python_symlinks(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / GENERATOR_PATH
    source.parent.mkdir(parents=True)
    source.symlink_to("target.py")
    _git(repo, "init")
    _git(repo, "config", "user.email", "review-check@example.test")
    _git(repo, "config", "user.name", "Review Check")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "symlink fixture")
    revision = _git(repo, "rev-parse", "HEAD")
    source.unlink()
    source.symlink_to("target . py")

    assert (
        review_check._non_executable_generator_drift_paths(
            repo,
            producer_revision=revision,
        )
        is None
    )


def test_non_executable_generator_drift_rejects_recorded_python_symlink(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / GENERATOR_PATH
    source.parent.mkdir(parents=True)
    source.symlink_to("VALUE = 1")
    _git(repo, "init")
    _git(repo, "config", "user.email", "review-check@example.test")
    _git(repo, "config", "user.name", "Review Check")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "recorded symlink fixture")
    revision = _git(repo, "rev-parse", "HEAD")
    source.unlink()
    source.write_text("VALUE = 1\n", encoding="utf-8")

    assert (
        review_check._non_executable_generator_drift_paths(
            repo,
            producer_revision=revision,
        )
        is None
    )


def test_non_executable_generator_drift_requires_producer_revision(
    tmp_path: Path,
) -> None:
    repo, _revision = _committed_fixture(tmp_path)

    with pytest.raises(
        review_check.ReviewRecipeCheckError,
        match="producer revision is unavailable",
    ):
        review_check._non_executable_generator_drift_paths(
            repo,
            producer_revision="f" * 40,
        )
