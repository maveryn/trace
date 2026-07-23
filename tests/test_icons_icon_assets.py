"""Tests for curated icon manifest loading."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.icons.shared.icon_assets import available_manifests, load_icon_manifest, resolve_icon_pool


def test_curated_icon_manifests_use_full_pools_only() -> None:
    assert set(available_manifests()) == {"all", "all_icons", "non_symmetry", "symmetry"}
    assert len(resolve_icon_pool("all_icons.txt")) == 3000
    assert len(resolve_icon_pool("all")) == 3000
    assert len(resolve_icon_pool("non_symmetry.txt")) == 2000
    assert len(resolve_icon_pool("symmetry.txt")) == 1000


@pytest.mark.parametrize(
    "manifest_name",
    [
        "icon-names.txt",
        "source_counts.json",
        "custom_subset.txt",
    ],
)
def test_curated_icon_manifest_loader_rejects_arbitrary_files(manifest_name: str) -> None:
    with pytest.raises(ValueError, match="unsupported icon manifest"):
        load_icon_manifest(manifest_name)
