from __future__ import annotations

import hashlib
import random
from collections import Counter

from trace_tasks.tasks.pages.shared.page_visual_assets import (
    available_page_visual_asset_roles,
    filter_page_visual_assets,
    load_page_visual_asset_manifest,
    load_page_visual_asset_sources,
    page_visual_asset_path,
    page_visual_asset_root,
    page_visual_asset_version,
    render_page_visual_asset_rgba,
    sample_page_visual_asset,
)
from trace_tasks.tasks.pages.shared.page_semantic_assets import (
    filter_page_semantic_assets,
    load_page_semantic_asset_manifest,
    page_semantic_asset_ids,
    page_semantic_asset_label,
    page_semantic_asset_manifest_metadata,
    render_page_semantic_asset_rgba,
)
from trace_tasks.tasks.pages.shared.page_text_resources import sample_page_context_batch, sample_page_label_batch


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_pages_visual_asset_manifest_counts_and_policy() -> None:
    assets = load_page_visual_asset_manifest()
    role_counts = Counter(role for asset in assets for role in asset.allowed_roles)

    assert len(assets) >= 300
    assert set(available_page_visual_asset_roles()) == {
        "hero_anchor",
        "section_illustration",
        "badge_spot",
    }
    assert 100 <= role_counts["hero_anchor"] <= 500
    assert 100 <= role_counts["section_illustration"] <= 500
    assert 100 <= role_counts["badge_spot"] <= 500
    assert {asset.license_spdx for asset in assets} <= {"Apache-2.0", "CC0-1.0", "MIT"}
    assert {asset.semantic_policy for asset in assets} == {"non_answer_visual_context"}
    assert all(asset.width_px > 0 and asset.height_px > 0 for asset in assets)
    assert all(0.01 <= asset.alpha_coverage <= 0.96 for asset in assets)


def test_pages_visual_asset_files_and_sources_are_valid() -> None:
    assets = load_page_visual_asset_manifest()
    asset_root = page_visual_asset_root()
    source_counts = Counter(asset.source_id for asset in assets)

    for asset in assets:
        assert (asset_root / asset.raw_path).exists()
        assert (asset_root / asset.local_license).exists()
        assert page_visual_asset_path(asset).exists()
        assert _sha256(page_visual_asset_path(asset)) == asset.normalized_checksum_sha256

    sources = load_page_visual_asset_sources()
    assert page_visual_asset_version() == sources["asset_version"]
    assert sources["asset_count"] == len(assets)
    assert sources["source_counts"] == dict(source_counts)


def test_pages_visual_asset_filtering_and_sampling_are_deterministic() -> None:
    hero_assets = filter_page_visual_assets(role="hero_anchor")
    monochrome_badges = filter_page_visual_assets(role="badge_spot", render_modes=("monochrome",))

    assert len(hero_assets) >= 100
    assert len(monochrome_badges) >= 100

    first = sample_page_visual_asset(random.Random(1234), role="hero_anchor")
    second = sample_page_visual_asset(random.Random(1234), role="hero_anchor")
    assert first.asset.asset_id == second.asset.asset_id
    assert first.role == "hero_anchor"
    assert first.candidate_count == len(hero_assets)
    assert first.to_metadata()["asset"]["asset_id"] == first.asset.asset_id


def test_pages_visual_asset_rendering_supports_color_and_tinted_monochrome() -> None:
    hero = sample_page_visual_asset(random.Random(7), role="hero_anchor")
    badge = sample_page_visual_asset(
        random.Random(7),
        role="badge_spot",
        render_modes=("monochrome",),
    )

    hero_image = render_page_visual_asset_rgba(hero.asset, size_px=(180, 120))
    badge_image = render_page_visual_asset_rgba(badge.asset, size_px=64, tint_rgb=(20, 80, 160))

    assert hero_image.mode == "RGBA"
    assert badge_image.mode == "RGBA"
    assert hero_image.width <= 180
    assert hero_image.height <= 120
    assert badge_image.width <= 64
    assert badge_image.height <= 64
    assert hero_image.getchannel("A").getbbox() is not None
    assert badge_image.getchannel("A").getbbox() is not None


def test_pages_semantic_asset_overlay_references_existing_page_assets() -> None:
    assets = load_page_semantic_asset_manifest()
    base_ids = {asset.asset_id for asset in load_page_visual_asset_manifest()}
    roles = Counter(asset.semantic_role for asset in assets)

    assert roles["metric_icon"] >= 10
    assert roles["marker"] >= 6
    assert {asset.asset_id for asset in assets} <= base_ids
    assert set(page_semantic_asset_ids(semantic_role="metric_icon", allowed_use="filter"))
    assert page_semantic_asset_label("calendar_icon") == "calendar icon"

    metadata = page_semantic_asset_manifest_metadata(semantic_role="marker", allowed_use="filter")
    assert metadata["semantic_policy"] == "answer_bearing_visual_overlay"
    assert "check_marker" in metadata["semantic_ids"]


def test_pages_semantic_asset_rendering_and_text_sampling_are_deterministic() -> None:
    first = render_page_semantic_asset_rgba("calendar_icon", size_px=48, tint_rgb=(20, 80, 160))
    second = render_page_semantic_asset_rgba("calendar_icon", size_px=48, tint_rgb=(20, 80, 160))
    assert first.size == second.size
    assert first.getchannel("A").getbbox() is not None

    marker_assets = filter_page_semantic_assets(semantic_role="marker", allowed_uses=("filter",))
    assert {asset.semantic_id for asset in marker_assets} >= {"check_marker", "star_marker"}

    label_batch = sample_page_label_batch(
        random.Random(11),
        role="test_label",
        count=4,
        manifest_name="categories/abstract_group_labels.txt",
        max_chars=14,
    )
    repeat_label_batch = sample_page_label_batch(
        random.Random(11),
        role="test_label",
        count=4,
        manifest_name="categories/abstract_group_labels.txt",
        max_chars=14,
    )
    assert label_batch.values == repeat_label_batch.values
    assert label_batch.to_metadata()["selections"][0]["manifest_path"].startswith("assets/labels/")

    context_batch = sample_page_context_batch(
        random.Random(13),
        role="test_context",
        count=2,
        manifest_names=("phrases/headlines.txt",),
    )
    assert len(set(context_batch.values)) == 2
    assert context_batch.to_metadata()["selections"][0]["manifest_path"].startswith("assets/context_text/")
