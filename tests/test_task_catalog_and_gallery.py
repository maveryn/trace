from __future__ import annotations

from collections import Counter
import json

from scripts import (
    generate_paper_domain_montage,
    generate_release_gallery,
    generate_task_catalog,
)
from trace_tasks.core.taxonomy import ACTIVE_DOMAINS


def test_generated_task_catalog_is_current_and_complete() -> None:
    catalog = generate_task_catalog.collect_catalog()
    outputs = generate_task_catalog.render_outputs(catalog)

    assert generate_task_catalog.check_outputs(outputs) == []
    assert catalog["schema_version"] == "trace_task_catalog_v1"
    assert catalog["summary"]["default_task_count"] == 1000
    assert catalog["summary"]["registered_task_count"] == 1000
    assert catalog["summary"]["domain_count"] == len(ACTIVE_DOMAINS) == 11
    assert len(catalog["tasks"]) == 1000
    assert {record["domain"] for record in catalog["tasks"]} == set(ACTIVE_DOMAINS)
    assert all(record["reasoning_operations"] for record in catalog["tasks"])
    assert all(
        len(record["program_contract_sha256"]) == 64 for record in catalog["tasks"]
    )
    domain_pages = {
        domain: outputs[generate_task_catalog.CATALOG_DIR / f"{domain}.md"].decode(
            "utf-8"
        )
        for domain in ACTIVE_DOMAINS
    }
    assert all(
        (
            f"[source]({generate_task_catalog.PUBLIC_SOURCE_BASE_URL}/"
            f"{record['source_path']})"
        )
        in domain_pages[record["domain"]]
        for record in catalog["tasks"]
    )
    assert all("[source](../../" not in page for page in domain_pages.values())


def test_release_gallery_is_exact_and_covers_every_domain() -> None:
    assert generate_release_gallery.check_gallery() == []

    manifest = json.loads(
        generate_release_gallery.MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "trace_release_gallery_v1"
    assert manifest["summary"] == {
        "domain_count": 11,
        "example_count": 22,
        "examples_per_domain": 2,
    }
    assert Counter(record["domain"] for record in manifest["examples"]) == {
        domain: 2 for domain in ACTIVE_DOMAINS
    }
    assert Counter(
        record["domain"] for record in manifest["examples"] if record["hero"]
    ) == {domain: 1 for domain in ACTIVE_DOMAINS}
    assert set(manifest["runtime"]) == {
        "native_libraries",
        "packages",
        "python",
    }
    assert {"cairo", "pillow_features"} == set(manifest["runtime"]["native_libraries"])
    assert {
        "cairosvg",
        "cairocffi",
        "cffi",
        "cssselect2",
        "defusedxml",
        "networkx",
        "numpy",
        "pillow",
        "pycparser",
        "pyyaml",
        "scipy",
        "tinycss2",
        "webencodings",
    } == set(manifest["runtime"]["packages"])
    assert (
        manifest["runtime"]["packages"]
        == generate_release_gallery._PINNED_RUNTIME_PACKAGES
    )
    assert (
        manifest["runtime"]["native_libraries"]["cairo"]
        == generate_release_gallery._PUBLIC_CAIRO_BASELINE
    )
    assert len(manifest["hero"]["task_ids"]) == 11


def test_paper_domain_montage_is_exact_and_covers_every_domain() -> None:
    assert generate_paper_domain_montage.check_montage() == []

    manifest = json.loads(
        generate_paper_domain_montage.MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "trace_paper_domain_montage_v1"
    assert manifest["selection_id"] == "trace_paper_domain_montage_v1"
    assert manifest["layout"] == {
        "panel_count": 11,
        "rendered_cell_count": 12,
        "rendered_row_counts": [4, 4, 4],
        "row_counts": [4, 4, 3],
    }
    assert manifest["brand_card"] == {
        "column": 3,
        "detail": "11 visual domains",
        "lettered": False,
        "mark_raster_path": "docs/assets/brand/trace-mark-light.png",
        "mark_raster_sha256": (
            "19fb18a6fc64881397629ae82c4ef8cda1b371ea09a12f5c26fac5bad02f4297"
        ),
        "mark_svg_path": "docs/assets/brand/trace-mark.svg",
        "mark_svg_sha256": (
            "6b85fc0b735b70339d56301e8a0067e6d5bbf729161a8aaace217f96ed347da3"
        ),
        "row": 2,
        "subtitle": "Grounded visual reasoning",
        "title": "Trace",
    }
    assert manifest["readme_header"] == {
        "mark": {
            "asset_path": "docs/assets/brand/trace-mark.svg",
            "asset_sha256": (
                "6b85fc0b735b70339d56301e8a0067e6d5bbf729161a8aaace217f96ed347da3"
            ),
            "display_height": 63,
            "display_width": 63,
        },
        "text": "Trace",
        "wordmark": {
            "asset_path": "docs/assets/brand/trace-wordmark.svg",
            "asset_sha256": (
                "f9c384df6b6f244d4a269be2461dcba8ff181160e42af7d7d75834903de77f5b"
            ),
            "display_height": 64,
            "display_width": 233,
        },
    }
    readme = (generate_paper_domain_montage.REPO_ROOT / "README.md").read_text(
        encoding="utf-8"
    )
    assert readme.startswith(
        '<h1 align="center">\n'
        '  <img src="docs/assets/brand/trace-mark.svg" alt="" '
        'width="63" height="63"><img '
        'src="docs/assets/brand/trace-wordmark.svg" alt="Trace" '
        'width="233" height="64">\n'
        "</h1>\n"
    )
    assert [record["domain"] for record in manifest["panels"]] == [
        item.domain for item in generate_paper_domain_montage.PANEL_SPECS
    ]
    assert {record["domain"] for record in manifest["panels"]} == set(ACTIVE_DOMAINS)
    assert len({record["task_id"] for record in manifest["panels"]}) == 11
    assert [record["panel_label"] for record in manifest["panels"]] == list(
        "abcdefghijk"
    )
    assert all(record["source_index"] == 0 for record in manifest["panels"])
    assert all(len(record["source_data_sha256"]) == 64 for record in manifest["panels"])
