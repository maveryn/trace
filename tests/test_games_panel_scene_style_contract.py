"""Games-domain regression tests for shared panel-scene treatments."""

from __future__ import annotations

from pathlib import Path

from trace_tasks.core.taxonomy import TASK_TAXONOMY
from trace_tasks.tasks.shared.visual_style.panel import PANEL_SCENE_TREATMENTS


REPO_ROOT = Path(__file__).resolve().parents[1]
GAMES_TASK_ROOT = REPO_ROOT / "src" / "trace_tasks" / "tasks" / "games"
GAMES_CONFIG_ROOT = (
    REPO_ROOT / "src" / "trace_tasks" / "resources" / "configs" / "domains" / "games"
)


def _scene_source(scene_id: str) -> str:
    scene_dir = GAMES_TASK_ROOT / scene_id
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(scene_dir.rglob("*.py"))
        if path.name != "__init__.py"
    )


def test_games_scenes_use_shared_panel_style() -> None:
    """Every public games scene should route through the shared panel style layer."""

    scenes = sorted(
        {
            entry.scene_id
            for entry in TASK_TAXONOMY.values()
            if entry.domain == "games"
        }
    )
    assert scenes
    missing: list[str] = []
    for scene_id in scenes:
        source = _scene_source(scene_id)
        if "resolve_game_panel_scene_style" not in source or "make_panel_scene_background" not in source:
            missing.append(str(scene_id))

    assert not missing, f"games scenes missing shared panel style resolver/background: {missing}"


def test_games_panel_style_uses_current_key_names_and_unrestricted_configs() -> None:
    """Games defaults should not keep legacy keys or narrow treatment availability."""

    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(GAMES_TASK_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    )
    assert "panel_treatment_weights" not in source_text
    assert "panel_palette_weights" not in source_text

    config_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(GAMES_CONFIG_ROOT.rglob("*.yaml"))
    )
    assert "panel_scene_treatments" not in config_text
    assert "panel_treatments" not in config_text


def test_games_shared_panel_treatment_count_is_25() -> None:
    """Games panel scenes inherit the global 25-treatment target."""

    assert len(PANEL_SCENE_TREATMENTS) == 25
