#!/usr/bin/env python3
"""Build the 11-domain README montage with the Trace mark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

# Make the source checkout importable without relying on an editable install or
# an ambient PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trace_tasks.core.source_layout_policy import parse_public_task_id
from trace_tasks.core.taxonomy import ACTIVE_DOMAINS, resolve_task_taxonomy
from trace_tasks.tasks.registry import list_default_task_ids

ASSET_ROOT = REPO_ROOT / "docs" / "assets" / "paper-domain-montage"
SOURCE_DIR = ASSET_ROOT / "sources"
OUTPUT_PATH = ASSET_ROOT / "trace-paper-domain-montage.png"
MANIFEST_PATH = REPO_ROOT / "docs" / "gallery" / "paper-domain-montage.v1.json"
BRAND_ROOT = REPO_ROOT / "docs" / "assets" / "brand"
BRAND_LOGO_SVG_PATH = BRAND_ROOT / "trace-logo.svg"
BRAND_MARK_SVG_PATH = BRAND_ROOT / "trace-mark.svg"
BRAND_MARK_RASTER_PATH = BRAND_ROOT / "trace-mark-light.png"
BRAND_WORDMARK_SVG_PATH = BRAND_ROOT / "trace-wordmark.svg"
FONT_PATH = (
    SRC_ROOT
    / "trace_tasks"
    / "resources"
    / "assets"
    / "fonts"
    / "google_fonts"
    / "noto_sans"
    / "NotoSans[wdth,wght].ttf"
)

SCHEMA_VERSION = "trace_paper_domain_montage_v1"
SELECTION_ID = "trace_paper_domain_montage_v1"
PAPER_ROW_COUNTS = (4, 4, 3)
RENDERED_ROW_COUNTS = (4, 4, 4)
CELL_SIZE = (360, 240)
IMAGE_HEIGHT = 202
GAP = 12
PADDING = 18
BACKGROUND = "#ffffff"
PANEL_BACKGROUND = "#f7f8fa"
PANEL_BORDER = "#cdd3db"
LABEL_COLOR = "#172033"
ACCENT_COLOR = "#3269c5"
BRAND_LOGO_SVG_SHA256 = (
    "41e2ec4642aca5f707adc06156db2c462e75478090a04add1cfa46fc4a2003da"
)
BRAND_MARK_SVG_SHA256 = (
    "6b85fc0b735b70339d56301e8a0067e6d5bbf729161a8aaace217f96ed347da3"
)
BRAND_MARK_RASTER_SHA256 = (
    "19fb18a6fc64881397629ae82c4ef8cda1b371ea09a12f5c26fac5bad02f4297"
)
BRAND_WORDMARK_SVG_SHA256 = (
    "f9c384df6b6f244d4a269be2461dcba8ff181160e42af7d7d75834903de77f5b"
)
BRAND_CARD_TITLE = "Trace"
BRAND_CARD_SUBTITLE_LINES = ("Grounded visual", "reasoning")
BRAND_CARD_SUBTITLE = " ".join(BRAND_CARD_SUBTITLE_LINES)
BRAND_CARD_DETAIL = "11 visual domains"
README_HEADER_MARK_HEIGHT = 63
README_HEADER_MARK_WIDTH = 63
README_HEADER_WORDMARK_HEIGHT = 64
README_HEADER_WORDMARK_WIDTH = 233


@dataclass(frozen=True)
class PanelSpec:
    """One committed example for the README montage."""

    label: str
    domain: str
    task_id: str
    query_id: str
    source_index: int
    instance_seed: int
    image_sha256: str
    source_data_sha256: str

    @property
    def scene_id(self) -> str:
        return parse_public_task_id(self.task_id).scene_id

    @property
    def source_path(self) -> Path:
        return SOURCE_DIR / f"{self.domain}.png"


PANEL_SPECS: tuple[PanelSpec, ...] = (
    PanelSpec(
        "Charts",
        "charts",
        "task_charts__multiseries__category_total_extremum_label",
        "largest_category_total_label",
        0,
        6971963630053741,
        "c745587a933839e2925d0e091e0876ba624594066a573fc279d85831e6d94f0c",
        "9e541ddfe6194e8973ea16457d2e8e36fd593e2a209df5b24f8ea2439fab8de1",
    ),
    PanelSpec(
        "Games",
        "games",
        "task_games__space_shooter__enemy_ship_count",
        "single",
        0,
        2618255346746328,
        "63fde69d84c704c212f2f438a3ad41dc489b2bc93ac1db3c3f17ba381815ef49",
        "722c72183b3d9b29972aa435a1427e35d2baa4ad3261b3fd0e4cb12ef0b3ac6f",
    ),
    PanelSpec(
        "Geometry",
        "geometry",
        "task_geometry__angle_relations__algebraic_angle_value",
        "single",
        0,
        6685141427653421,
        "1dc26906e16ee63df8b78347348cd47e766e40248a9272926c8af62cf98329b0",
        "c3b12e974c3a3543e84f785061e9da09b35fa55dd8b94fff827a26fdd7df44fb",
    ),
    PanelSpec(
        "Graphs",
        "graph",
        "task_graph__node_link__shortest_path_length",
        "directed_shortest_path_length",
        0,
        2481365412104608,
        "c5aaa80bb5f3cd91d01b347908459482f8eb7f7e8727634ba5c26977da431251",
        "fb10fe9fe7dc784b7025a49ce16b8e28b2fc5e19b182d00d7726cc67916a6da5",
    ),
    PanelSpec(
        "Icons",
        "icons",
        "task_icons__icon_field__most_frequent_type_count",
        "single",
        0,
        8191844928700627,
        "d5c5bfb8d3a3a78dcff81e1119f1e887f6177ee61484a717c3c2fd08859ef441",
        "d9ee0d8eee20037a8534516f682a5de0cdf72c9bbf40cc0c04f58f8e75b03210",
    ),
    PanelSpec(
        "Illustrations",
        "illustrations",
        "task_illustrations__park_playground__playground_equipment_count",
        "single",
        0,
        261162083042051,
        "c42eec98507f59adceb1bc3c7dc6ec7335a97b375e6e0874ab03b25778f71b7b",
        "59abc40bb38d6065aeb0adceb7ee93f6843dc69823732d2dc3f7dc828e5d1c09",
    ),
    PanelSpec(
        "Pages",
        "pages",
        "task_pages__record_table__value_threshold_in_group_count",
        "single",
        0,
        3621091651242047,
        "d87de3758165fbf155a48331e80554a8e5e48fd077b577dc7f4edcad61a14354",
        "f24b689deda9fe25df892144af49f65378acf7b1e1c27f7fa3fc0a45d20f2507",
    ),
    PanelSpec(
        "Physics",
        "physics",
        "task_physics__free_body_forces__net_force_direction_choice",
        "single",
        0,
        8791560635313457,
        "85f979e35f6958b51e8a778824daa429773dccc0c83da0ff65816ae5d23b1723",
        "1520372de7b82f2d2906c9ded390b3b54e50ce9734555cf01e40fbd49c23813d",
    ),
    PanelSpec(
        "Puzzles",
        "puzzles",
        "task_puzzles__raven_matrix__raven_count_progression_label",
        "single",
        0,
        8979951069889835,
        "1979f2ce6e43ada5fd4dbea3f736863e29fd07b2915d12c6825b12c3b684e7e6",
        "a6ee8053568831f93bca60caa872ee6516a66d2a7b992faf97fd0a32549e4f78",
    ),
    PanelSpec(
        "Symbolic",
        "symbolic",
        "task_symbolic__clock__equivalent_time_label",
        "analog_reference_digital_options",
        0,
        5401932260136482,
        "cdf577e8d8c9f870c4c958aa3638837ea26991185b28aef5108df3642027f7e2",
        "2ae221be0ee613e217216ccf87fd35ab109b6d9b092d13df6a3fd186053cf447",
    ),
    PanelSpec(
        "3D scenes",
        "three_d",
        "task_three_d__object_scene__camera_depth_relation_count",
        "closer_to_camera_than_reference_count",
        0,
        948033558578505,
        "59e5c8a206c0932509134b16bdf750a33e7c73409b3eb04d7557ed6656bf77ee",
        "07e12ddee020980e4460ee9eab63aea827cd7346cb39bcda9817d56fc6bca6aa",
    ),
)


class MontageError(RuntimeError):
    """Raised when the frozen paper montage inputs are invalid."""


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _validate_and_load_sources() -> dict[str, Image.Image]:
    if len(PANEL_SPECS) != sum(PAPER_ROW_COUNTS):
        raise MontageError("paper montage must contain exactly eleven panels")
    if len(PANEL_SPECS) + 1 != sum(RENDERED_ROW_COUNTS):
        raise MontageError("rendered montage must contain eleven panels and one card")
    if len(set(RENDERED_ROW_COUNTS)) != 1:
        raise MontageError("rendered montage rows must form a rectangular grid")
    if len({item.task_id for item in PANEL_SPECS}) != len(PANEL_SPECS):
        raise MontageError("paper montage task ids must be unique")
    if {item.domain for item in PANEL_SPECS} != set(ACTIVE_DOMAINS):
        raise MontageError(
            "paper montage must contain every public domain exactly once"
        )

    default_ids = set(list_default_task_ids())
    expected_paths = {item.source_path for item in PANEL_SPECS}
    actual_paths = set(SOURCE_DIR.iterdir()) if SOURCE_DIR.is_dir() else set()
    if actual_paths != expected_paths:
        missing = sorted(_repo_path(path) for path in expected_paths - actual_paths)
        unexpected = sorted(_repo_path(path) for path in actual_paths - expected_paths)
        raise MontageError(
            f"paper montage source inventory differs; missing={missing}, "
            f"unexpected={unexpected}"
        )

    images: dict[str, Image.Image] = {}
    for item in PANEL_SPECS:
        if item.task_id not in default_ids:
            raise MontageError(f"paper montage task is not active: {item.task_id}")
        taxonomy = resolve_task_taxonomy(item.task_id)
        if taxonomy.domain != item.domain or taxonomy.scene_id != item.scene_id:
            raise MontageError(f"paper montage taxonomy disagrees for {item.task_id}")
        if item.source_index < 0 or item.instance_seed < 0 or not item.query_id:
            raise MontageError(f"paper montage identity is invalid for {item.task_id}")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (item.image_sha256, item.source_data_sha256)
        ):
            raise MontageError(f"paper montage hash is invalid for {item.task_id}")
        if _sha256_path(item.source_path) != item.image_sha256:
            raise MontageError(f"paper montage source hash differs for {item.task_id}")

        with Image.open(item.source_path) as source:
            if source.format != "PNG" or source.mode != "RGB":
                raise MontageError(
                    f"paper montage source must be an RGB PNG: {item.task_id}"
                )
            if source.info:
                raise MontageError(
                    f"paper montage source contains PNG metadata: {item.task_id}"
                )
            source.load()
            images[item.domain] = source.copy()
    return images


def _validate_and_load_brand_mark() -> Image.Image:
    expected = (
        (BRAND_LOGO_SVG_PATH, BRAND_LOGO_SVG_SHA256),
        (BRAND_MARK_SVG_PATH, BRAND_MARK_SVG_SHA256),
        (BRAND_MARK_RASTER_PATH, BRAND_MARK_RASTER_SHA256),
        (BRAND_WORDMARK_SVG_PATH, BRAND_WORDMARK_SVG_SHA256),
    )
    for path, expected_sha256 in expected:
        if not path.is_file():
            raise MontageError(f"missing brand asset: {_repo_path(path)}")
        if _sha256_path(path) != expected_sha256:
            raise MontageError(f"brand asset hash differs: {_repo_path(path)}")

    with Image.open(BRAND_MARK_RASTER_PATH) as source:
        if source.format != "PNG" or source.mode != "RGBA":
            raise MontageError("brand-card raster must be an RGBA PNG")
        if source.size != (280, 280):
            raise MontageError("brand-card raster must be 280x280 pixels")
        if source.info:
            raise MontageError("brand-card raster contains PNG metadata")
        source.load()
        return source.copy()


def _load_font(size: int, variation: str = "Regular") -> ImageFont.FreeTypeFont:
    if not FONT_PATH.is_file():
        raise MontageError(f"missing packaged montage font: {_repo_path(FONT_PATH)}")
    font = ImageFont.truetype(str(FONT_PATH), size=size)
    font.set_variation_by_name(variation)
    return font


def _fit_on_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, PANEL_BACKGROUND)
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def _render_brand_card(mark: Image.Image) -> Image.Image:
    cell_width, cell_height = CELL_SIZE
    card = Image.new("RGB", CELL_SIZE, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(card)

    mark_panel = ImageOps.contain(mark, (180, 180), Image.Resampling.LANCZOS)
    mark_x = 6 + (180 - mark_panel.width) // 2
    mark_y = (cell_height - mark_panel.height) // 2
    card.paste(mark_panel, (mark_x, mark_y), mark_panel)

    title_font = _load_font(43, "Bold")
    subtitle_font = _load_font(17)
    detail_font = _load_font(16, "SemiBold")
    text_x = 183
    draw.text((text_x, 56), BRAND_CARD_TITLE, font=title_font, fill=LABEL_COLOR)
    for line_index, line in enumerate(BRAND_CARD_SUBTITLE_LINES):
        draw.text(
            (text_x, 116 + 23 * line_index),
            line,
            font=subtitle_font,
            fill=LABEL_COLOR,
        )
    draw.text((text_x, 174), BRAND_CARD_DETAIL, font=detail_font, fill=ACCENT_COLOR)
    draw.rectangle(
        (0, 0, cell_width - 1, cell_height - 1),
        outline=PANEL_BORDER,
        width=1,
    )
    return card


def _render_montage(
    images: Mapping[str, Image.Image], brand_mark: Image.Image
) -> tuple[bytes, tuple[int, int]]:
    columns = max(RENDERED_ROW_COUNTS)
    cell_width, cell_height = CELL_SIZE
    width = PADDING * 2 + columns * cell_width + (columns - 1) * GAP
    height = (
        PADDING * 2
        + len(RENDERED_ROW_COUNTS) * cell_height
        + (len(RENDERED_ROW_COUNTS) - 1) * GAP
    )
    montage = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(montage)
    font = _load_font(19)

    for item_index, item in enumerate(PANEL_SPECS):
        row_index, column_index = divmod(item_index, columns)
        y = PADDING + row_index * (cell_height + GAP)
        x = PADDING + column_index * (cell_width + GAP)
        panel_label = chr(ord("a") + item_index)

        image_panel = _fit_on_panel(
            images[item.domain], (cell_width - 4, IMAGE_HEIGHT - 2)
        )
        montage.paste(image_panel, (x + 2, y + 2))
        draw.line(
            (x + 1, y + IMAGE_HEIGHT, x + cell_width - 2, y + IMAGE_HEIGHT),
            fill=PANEL_BORDER,
            width=1,
        )
        text = f"({panel_label}) {item.label}"
        box = draw.textbbox((0, 0), text, font=font)
        text_x = x + (cell_width - (box[2] - box[0])) // 2
        text_y = (
            y
            + IMAGE_HEIGHT
            + (cell_height - IMAGE_HEIGHT - (box[3] - box[1])) // 2
            - box[1]
        )
        draw.text((text_x, text_y), text, font=font, fill=LABEL_COLOR)
        draw.rectangle(
            (x, y, x + cell_width - 1, y + cell_height - 1),
            outline=PANEL_BORDER,
            width=1,
        )

    brand_index = len(PANEL_SPECS)
    brand_row, brand_column = divmod(brand_index, columns)
    if brand_row >= len(RENDERED_ROW_COUNTS):
        raise MontageError("brand card does not fit the rendered montage grid")
    brand_card = _render_brand_card(brand_mark)
    montage.paste(
        brand_card,
        (
            PADDING + brand_column * (cell_width + GAP),
            PADDING + brand_row * (cell_height + GAP),
        ),
    )

    output = BytesIO()
    montage.save(output, format="PNG", optimize=True, compress_level=9)
    return output.getvalue(), montage.size


def generate_montage() -> tuple[dict[str, Any], bytes]:
    """Validate the paper inputs and render their public README montage."""

    images = _validate_and_load_sources()
    brand_mark = _validate_and_load_brand_mark()
    output, size = _render_montage(images, brand_mark)
    panels: list[dict[str, Any]] = []
    for order, item in enumerate(PANEL_SPECS):
        image = images[item.domain]
        panels.append(
            {
                "domain": item.domain,
                "height": image.height,
                "instance_seed": item.instance_seed,
                "label": item.label,
                "order": order,
                "panel_label": chr(ord("a") + order),
                "query_id": item.query_id,
                "scene_id": item.scene_id,
                "source_data_sha256": item.source_data_sha256,
                "source_image_path": _repo_path(item.source_path),
                "source_image_sha256": item.image_sha256,
                "source_index": item.source_index,
                "task_id": item.task_id,
                "width": image.width,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_id": SELECTION_ID,
        "generator": {
            "path": _repo_path(Path(__file__).resolve()),
            "sha256": _sha256_path(Path(__file__).resolve()),
        },
        "layout": {
            "panel_count": len(PANEL_SPECS),
            "rendered_cell_count": sum(RENDERED_ROW_COUNTS),
            "rendered_row_counts": list(RENDERED_ROW_COUNTS),
            "row_counts": list(PAPER_ROW_COUNTS),
        },
        "brand_card": {
            "column": len(PANEL_SPECS) % max(RENDERED_ROW_COUNTS),
            "detail": BRAND_CARD_DETAIL,
            "lettered": False,
            "mark_raster_path": _repo_path(BRAND_MARK_RASTER_PATH),
            "mark_raster_sha256": BRAND_MARK_RASTER_SHA256,
            "mark_svg_path": _repo_path(BRAND_MARK_SVG_PATH),
            "mark_svg_sha256": BRAND_MARK_SVG_SHA256,
            "row": len(PANEL_SPECS) // max(RENDERED_ROW_COUNTS),
            "subtitle": BRAND_CARD_SUBTITLE,
            "title": BRAND_CARD_TITLE,
        },
        "readme_header": {
            "mark": {
                "asset_path": _repo_path(BRAND_MARK_SVG_PATH),
                "asset_sha256": BRAND_MARK_SVG_SHA256,
                "display_height": README_HEADER_MARK_HEIGHT,
                "display_width": README_HEADER_MARK_WIDTH,
            },
            "text": "Trace",
            "wordmark": {
                "asset_path": _repo_path(BRAND_WORDMARK_SVG_PATH),
                "asset_sha256": BRAND_WORDMARK_SVG_SHA256,
                "display_height": README_HEADER_WORDMARK_HEIGHT,
                "display_width": README_HEADER_WORDMARK_WIDTH,
            },
        },
        "montage": {
            "height": size[1],
            "image_path": _repo_path(OUTPUT_PATH),
            "image_sha256": _sha256_bytes(output),
            "width": size[0],
        },
        "provenance": {
            "paper_figure_name": "domain_montage.pdf",
            "paper_source_repository_head": (
                "64f903a52b47b5330f1ada1645d7868535075e72"
            ),
            "paper_figure_sha256": (
                "62c6c14c1213772742d3ab10eda6a53b46773ca22e8a1164f0cfcfb1afc28bec"
            ),
            "pillow_version": metadata.version("Pillow"),
            "composite_policy": (
                "One example from each Trace domain, followed by the Trace mark."
            ),
            "source_policy": "Committed PNG inputs for the README domain montage.",
        },
        "panels": panels,
    }
    return manifest, output


def check_montage() -> list[str]:
    """Validate the inputs and byte-compare generated outputs."""

    try:
        manifest, output = generate_montage()
    except (MontageError, OSError, ValueError) as exc:
        return [f"README montage validation failed: {exc}"]

    expected_manifest = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    problems: list[str] = []
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_bytes() != output:
        problems.append(f"stale or missing: {_repo_path(OUTPUT_PATH)}")
    if not MANIFEST_PATH.is_file() or MANIFEST_PATH.read_bytes() != expected_manifest:
        problems.append(f"stale or missing: {_repo_path(MANIFEST_PATH)}")

    serialized = expected_manifest.decode("utf-8")
    forbidden_prefixes = (
        "/" + "home" + "/",
        "/" + "dev" + "/" + "shm",
        "/" + "workspace" + "/",
        "file" + "://",
    )
    for forbidden in forbidden_prefixes:
        if forbidden in serialized:
            problems.append(f"README montage manifest contains {forbidden!r}")
    return problems


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the 11-domain README montage")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed inputs or outputs differ",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        problems = check_montage()
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            print(
                "paper domain montage is stale; run "
                "`python scripts/generate_paper_domain_montage.py`",
                file=sys.stderr,
            )
            return 1
        print("paper domain montage is up to date")
        return 0

    try:
        manifest, output = generate_montage()
    except (MontageError, OSError, ValueError) as exc:
        print(f"paper domain montage generation failed: {exc}", file=sys.stderr)
        return 1
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(output)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {_repo_path(OUTPUT_PATH)}")
    print(f"wrote {_repo_path(MANIFEST_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
