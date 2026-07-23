from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks import generate_task
from trace_tasks.resources import resource_path, resource_root, safe_resource_join
from trace_tasks.tasks.pages.shared.page_visual_assets import (
    load_page_visual_asset_manifest,
    page_visual_asset_path,
)
from trace_tasks.tasks.shared.font_assets import list_font_families, resolve_font_paths


def test_resource_tree_contains_runtime_inputs() -> None:
    root = resource_root()
    assert root.is_dir()
    assert resource_path("assets", "icons", "all_icons.txt").is_file()
    assert resource_path("configs", "domains", "geometry", "base.yaml").is_file()
    assert resource_path(
        "prompts",
        "geometry",
        "graph_paper",
        "geometry_graph_paper_v1.json",
    ).is_file()


@pytest.mark.parametrize(
    "parts",
    [
        ("../outside",),
        ("assets", "..", "outside"),
        (str(Path("/") / "tmp" / "outside"),),
    ],
)
def test_resource_paths_reject_traversal(parts: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        resource_path(*parts)
    with pytest.raises(ValueError):
        safe_resource_join(resource_root(), *parts)


def test_asset_backed_tasks_generate_outside_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    outputs = [
        generate_task("task_icons__named_grid__line_adjacency_pair_count", seed=7),
        generate_task("task_pages__calendar__date_weekday_label", seed=7),
        generate_task("task_charts__region_map__adjacent_category_count", seed=7),
    ]
    assert all(output.image.width > 0 and output.image.height > 0 for output in outputs)


def test_page_visual_and_font_resources_load() -> None:
    page_assets = load_page_visual_asset_manifest()
    assert page_assets
    assert page_visual_asset_path(page_assets[0]).is_file()

    font_families = list_font_families()
    assert font_families
    assert resolve_font_paths(font_families[0], bold=False)
