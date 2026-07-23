#!/usr/bin/env python3
"""Generate the deterministic public Trace gallery and README hero montage.

Only public task APIs and packaged resources are used.  The output manifest
records content-addressed provenance instead of a checkout-local path or a git
revision that does not yet contain the generated artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata
from io import BytesIO
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence

import cairocffi
from PIL import Image, ImageDraw, ImageFont, ImageOps, features

# Make the source checkout importable without relying on an editable install or
# an ambient PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trace_tasks.core.source_layout_policy import parse_public_task_id
from trace_tasks.core.taxonomy import ACTIVE_DOMAINS, resolve_task_taxonomy
from trace_tasks.tasks.registry import create_task, list_default_task_ids

GALLERY_DIR = REPO_ROOT / "docs" / "assets" / "gallery"
MANIFEST_PATH = REPO_ROOT / "docs" / "gallery" / "manifest.v1.json"
CATALOG_PATH = REPO_ROOT / "docs" / "task_catalog" / "catalog.v1.json"
HERO_PATH = GALLERY_DIR / "trace-hero-montage.png"
SOURCE_ROOT = REPO_ROOT / "src" / "trace_tasks"
FONT_PATH = (
    SOURCE_ROOT
    / "resources"
    / "assets"
    / "fonts"
    / "google_fonts"
    / "noto_sans"
    / "NotoSans[wdth,wght].ttf"
)

GALLERY_SCHEMA_VERSION = "trace_release_gallery_v1"
GALLERY_GENERATOR_VERSION = "v1"
MAX_IMAGE_SIZE = (1200, 900)
MAX_ATTEMPTS = 100
HERO_COLUMNS = 4
HERO_CELL_SIZE = (360, 240)
HERO_GAP = 12
HERO_PADDING = 18
_RUNTIME_DISTRIBUTIONS: tuple[str, ...] = (
    "CairoSVG",
    "cairocffi",
    "cffi",
    "cssselect2",
    "defusedxml",
    "networkx",
    "numpy",
    "Pillow",
    "pycparser",
    "PyYAML",
    "scipy",
    "tinycss2",
    "webencodings",
)
_PINNED_RUNTIME_PACKAGES: dict[str, str] = {
    "cairosvg": "2.9.0",
    "cairocffi": "1.7.1",
    "cffi": "2.1.0",
    "cssselect2": "0.9.0",
    "defusedxml": "0.7.1",
    "networkx": "3.4.2",
    "numpy": "2.2.6",
    "pillow": "12.2.0",
    "pycparser": "3.0",
    "pyyaml": "6.0.3",
    "scipy": "1.15.3",
    "tinycss2": "1.5.1",
    "webencodings": "0.5.1",
}
_PUBLIC_CAIRO_BASELINE = "1.18.0"
_PILLOW_NATIVE_FEATURES: tuple[str, ...] = (
    "freetype2",
    "littlecms2",
    "webp",
    "avif",
    "raqm",
    "fribidi",
    "harfbuzz",
    "libjpeg_turbo",
    "zlib_ng",
    "jpg_2000",
    "zlib",
    "libtiff",
)


@dataclass(frozen=True)
class GallerySelection:
    """One reviewed, deterministic public gallery example."""

    domain: str
    task_id: str
    seed: int
    caption: str
    hero: bool = False

    @property
    def scene_id(self) -> str:
        return parse_public_task_id(self.task_id).scene_id

    @property
    def objective_contract(self) -> str:
        return parse_public_task_id(self.task_id).objective_contract

    @property
    def filename(self) -> str:
        return f"{self.domain}-{self.scene_id}-{self.objective_contract}.png"


GALLERY_SELECTIONS: tuple[GallerySelection, ...] = (
    GallerySelection(
        "charts",
        "task_charts__dashboard__category_total_extremum_label",
        3201,
        "Mixed-chart dashboard",
    ),
    GallerySelection(
        "charts",
        "task_charts__radial_sankey__dominant_endpoint_label",
        3202,
        "Radial Sankey routing",
        hero=True,
    ),
    GallerySelection(
        "games",
        "task_games__chess__colored_piece_kind_count",
        3301,
        "Chess-board counting",
    ),
    GallerySelection(
        "games",
        "task_games__pacman__route_score_value",
        3302,
        "Arcade route reasoning",
        hero=True,
    ),
    GallerySelection(
        "geometry",
        "task_geometry__graph_paper__polygon_area_value",
        3401,
        "Coordinate geometry",
    ),
    GallerySelection(
        "geometry",
        "task_geometry__solid_revolution__revolution_double_cone_volume_value",
        3402,
        "Solid of revolution",
        hero=True,
    ),
    GallerySelection(
        "graph",
        "task_graph__node_link__shortest_path_length",
        3501,
        "Node-link shortest path",
    ),
    GallerySelection(
        "graph",
        "task_graph__metro__shortest_path_length",
        3502,
        "Metro network reasoning",
        hero=True,
    ),
    GallerySelection(
        "icons",
        "task_icons__mirror_grid__missing_mirror_cell_label",
        3601,
        "Mirror-grid completion",
        hero=True,
    ),
    GallerySelection(
        "icons",
        "task_icons__wallpaper_panels__same_pattern_as_reference_label",
        3602,
        "Wallpaper pattern matching",
    ),
    GallerySelection(
        "illustrations",
        "task_illustrations__isometric_farmstead__terrain_elevation_extremum_label",
        3701,
        "Isometric terrain reasoning",
    ),
    GallerySelection(
        "illustrations",
        "task_illustrations__pixel_village__territory_object_count",
        3702,
        "Pixel-art spatial counting",
        hero=True,
    ),
    GallerySelection(
        "symbolic",
        "task_symbolic__chemical_equation__balanced_option_label",
        3801,
        "Chemical equation balancing",
        hero=True,
    ),
    GallerySelection(
        "symbolic",
        "task_symbolic__clock__full_time_readout",
        3802,
        "Analog clock readout",
    ),
    GallerySelection(
        "pages",
        "task_pages__map__destination_after_directions_label",
        3901,
        "Map direction following",
        hero=True,
    ),
    GallerySelection(
        "pages",
        "task_pages__timeline__relative_position_event_label",
        3902,
        "Timeline relations",
    ),
    GallerySelection(
        "physics",
        "task_physics__ray_optics__ray_bounce_count",
        4001,
        "Ray-optics reasoning",
    ),
    GallerySelection(
        "physics",
        "task_physics__circuit_state_change__bulb_brightness_change_label",
        4002,
        "Circuit state change",
        hero=True,
    ),
    GallerySelection(
        "puzzles",
        "task_puzzles__sudoku__marked_cell_value",
        4101,
        "Sudoku deduction",
    ),
    GallerySelection(
        "puzzles",
        "task_puzzles__polyomino_assembly__composition_result_label",
        4102,
        "Polyomino composition",
        hero=True,
    ),
    GallerySelection(
        "three_d",
        "task_three_d__room__wall_object_camera_distance_label",
        4201,
        "Room-camera geometry",
    ),
    GallerySelection(
        "three_d",
        "task_three_d__carousel__between_object_type_anchors_count",
        4202,
        "3D carousel relations",
        hero=True,
    ),
)


class GalleryError(RuntimeError):
    """Raised when gallery generation or freshness validation fails."""


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_tree_sha256() -> str:
    """Hash all public runtime inputs using repository-relative paths."""

    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in SOURCE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    """Record the Python and native versions that can affect raster output."""

    return {
        "native_libraries": {
            "cairo": cairocffi.cairo_version_string(),
            "pillow_features": {
                name: features.version(name) for name in _PILLOW_NATIVE_FEATURES
            },
        },
        "packages": {
            distribution.lower(): metadata.version(distribution)
            for distribution in _RUNTIME_DISTRIBUTIONS
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def _validate_committed_runtime(raw: Any) -> list[str]:
    """Validate byte provenance without requiring the checker to share it."""

    problems: list[str] = []
    if not isinstance(raw, Mapping):
        return ["gallery runtime provenance is missing"]
    if set(raw) != {"native_libraries", "packages", "python"}:
        problems.append("gallery runtime provenance fields disagree")

    packages = raw.get("packages")
    if packages != _PINNED_RUNTIME_PACKAGES:
        problems.append("gallery runtime packages do not match constraints/release.txt")

    native = raw.get("native_libraries")
    if not isinstance(native, Mapping):
        problems.append("gallery native-library provenance is missing")
    else:
        if native.get("cairo") != _PUBLIC_CAIRO_BASELINE:
            problems.append(
                f"gallery Cairo provenance must match Ubuntu 24.04 baseline {_PUBLIC_CAIRO_BASELINE}"
            )
        pillow_features = native.get("pillow_features")
        if not isinstance(pillow_features, Mapping) or set(pillow_features) != set(
            _PILLOW_NATIVE_FEATURES
        ):
            problems.append("gallery Pillow native-feature provenance fields disagree")
        elif not all(
            value is None or isinstance(value, str)
            for value in pillow_features.values()
        ):
            problems.append(
                "gallery Pillow native-feature versions must be strings or null"
            )

    python = raw.get("python")
    if not isinstance(python, Mapping):
        problems.append("gallery Python provenance is missing")
    else:
        if python.get("implementation") != "CPython":
            problems.append("gallery Python provenance must identify CPython")
        version = python.get("version")
        if (
            not isinstance(version, str)
            or re.fullmatch(r"3\.(?:10|11|12)\.\d+", version) is None
        ):
            problems.append(
                "gallery Python provenance is outside the supported 3.10-3.12 range"
            )
    return problems


def _task_doc_path(task_id: str) -> Path:
    parts = parse_public_task_id(task_id)
    return (
        REPO_ROOT / "docs" / "tasks" / parts.domain / parts.scene_id / f"{task_id}.md"
    )


def _task_source_path(task_id: str) -> Path:
    parts = parse_public_task_id(task_id)
    return (
        SOURCE_ROOT
        / "tasks"
        / parts.domain
        / parts.scene_id
        / f"{parts.objective_contract}.py"
    )


def _validate_selections() -> None:
    default_ids = set(list_default_task_ids())
    if len(GALLERY_SELECTIONS) != len(set(item.task_id for item in GALLERY_SELECTIONS)):
        raise GalleryError("gallery task ids must be unique")
    if any(item.seed < 0 for item in GALLERY_SELECTIONS):
        raise GalleryError("gallery seeds must be non-negative")

    counts = {domain: 0 for domain in ACTIVE_DOMAINS}
    hero_counts = {domain: 0 for domain in ACTIVE_DOMAINS}
    for item in GALLERY_SELECTIONS:
        if item.task_id not in default_ids:
            raise GalleryError(f"gallery task is not active: {item.task_id}")
        taxonomy = resolve_task_taxonomy(item.task_id)
        if taxonomy.domain != item.domain:
            raise GalleryError(f"gallery domain disagrees for {item.task_id}")
        counts[item.domain] += 1
        hero_counts[item.domain] += int(item.hero)
    if any(counts[domain] < 2 for domain in ACTIVE_DOMAINS):
        raise GalleryError(
            f"gallery does not include two examples per domain: {counts}"
        )
    if any(hero_counts[domain] != 1 for domain in ACTIVE_DOMAINS):
        raise GalleryError(
            f"hero montage must select exactly one task per domain: {hero_counts}"
        )


def _optimized_png(image: Image.Image) -> tuple[bytes, tuple[int, int]]:
    """Return a legible, bounded, metadata-free PNG."""

    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB")
    rgb.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    output = BytesIO()
    rgb.save(output, format="PNG", optimize=True, compress_level=9)
    return output.getvalue(), rgb.size


def _prompt_provenance(trace_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract safe, package-relative prompt provenance without prompt text."""

    candidates: list[Mapping[str, Any]] = []
    query_spec = trace_payload.get("query_spec")
    if isinstance(query_spec, Mapping):
        candidates.append(query_spec)
        prompt_variant = query_spec.get("prompt_variant")
        if isinstance(prompt_variant, Mapping):
            candidates.insert(0, prompt_variant)
    render_spec = trace_payload.get("render_spec")
    if isinstance(render_spec, Mapping):
        prompt_spec = render_spec.get("prompt")
        if isinstance(prompt_spec, Mapping):
            candidates.append(prompt_spec)

    allowed_keys = (
        "prompt_bundle_id",
        "prompt_bundle_path",
        "prompt_bundle_hash",
        "prompt_schema_version",
        "schema_version",
        "selected_indices",
        "selected_keys",
    )
    for candidate in candidates:
        if "prompt_bundle_id" not in candidate:
            continue
        return {key: candidate[key] for key in allowed_keys if key in candidate}
    return {}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def _fit_on_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "#f6f7fa")
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def _title_tile(size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, "#132544")
    draw = ImageDraw.Draw(tile)
    accent = "#67e8f9"
    draw.rounded_rectangle((22, 22, 84, 30), radius=4, fill=accent)
    draw.text((22, 52), "TRACE", font=_load_font(50), fill="white")
    draw.text(
        (24, 122),
        "1,000 tasks\n11 visual domains",
        font=_load_font(23),
        fill="#d9e8ff",
        spacing=5,
    )
    return tile


def _hero_montage(
    images_by_task: Mapping[str, Image.Image],
) -> tuple[bytes, tuple[int, int]]:
    hero_items = [item for item in GALLERY_SELECTIONS if item.hero]
    hero_by_domain = {item.domain: item for item in hero_items}
    ordered: list[GallerySelection | None] = [None]
    ordered.extend(hero_by_domain[domain] for domain in ACTIVE_DOMAINS)

    rows = (len(ordered) + HERO_COLUMNS - 1) // HERO_COLUMNS
    cell_width, cell_height = HERO_CELL_SIZE
    width = HERO_PADDING * 2 + HERO_COLUMNS * cell_width + (HERO_COLUMNS - 1) * HERO_GAP
    height = HERO_PADDING * 2 + rows * cell_height + (rows - 1) * HERO_GAP
    montage = Image.new("RGB", (width, height), "#07111f")
    label_font = _load_font(19)

    for index, item in enumerate(ordered):
        column = index % HERO_COLUMNS
        row = index // HERO_COLUMNS
        x = HERO_PADDING + column * (cell_width + HERO_GAP)
        y = HERO_PADDING + row * (cell_height + HERO_GAP)
        if item is None:
            tile = _title_tile(HERO_CELL_SIZE)
        else:
            tile = Image.new("RGB", HERO_CELL_SIZE, "#f6f7fa")
            image_height = cell_height - 34
            panel = _fit_on_panel(
                images_by_task[item.task_id], (cell_width, image_height)
            )
            tile.paste(panel, (0, 0))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, image_height, cell_width, cell_height), fill="#132544")
            draw.text(
                (12, image_height + 5),
                item.domain.replace("_", " "),
                font=label_font,
                fill="white",
            )
        montage.paste(tile, (x, y))

    output = BytesIO()
    montage.save(output, format="PNG", optimize=True, compress_level=9)
    return output.getvalue(), montage.size


def generate_gallery() -> tuple[dict[str, Any], dict[Path, bytes]]:
    """Generate all images and their content-addressed manifest."""

    _validate_selections()
    if not CATALOG_PATH.is_file():
        raise GalleryError(
            "missing docs/task_catalog/catalog.v1.json; generate the task catalog first"
        )

    outputs: dict[Path, bytes] = {}
    entries: list[dict[str, Any]] = []
    images_by_task: dict[str, Image.Image] = {}
    for order, selection in enumerate(GALLERY_SELECTIONS):
        generated = create_task(selection.task_id).generate(
            selection.seed,
            params={},
            max_attempts=MAX_ATTEMPTS,
        )
        source_size = tuple(int(value) for value in generated.image.size)
        png_bytes, size = _optimized_png(generated.image)
        output_path = GALLERY_DIR / selection.filename
        outputs[output_path] = png_bytes
        images_by_task[selection.task_id] = Image.open(BytesIO(png_bytes)).convert(
            "RGB"
        )

        doc_path = _task_doc_path(selection.task_id)
        source_path = _task_source_path(selection.task_id)
        entries.append(
            {
                "annotation_type": generated.annotation_gt.type,
                "answer_type": generated.answer_gt.type,
                "caption": selection.caption,
                "domain": selection.domain,
                "height": size[1],
                "hero": selection.hero,
                "image_id": generated.image_id,
                "image_path": _repo_path(output_path),
                "image_sha256": _sha256_bytes(png_bytes),
                "max_attempts": MAX_ATTEMPTS,
                "objective_contract": selection.objective_contract,
                "order": order,
                "params": {},
                "prompt": _prompt_provenance(generated.trace_payload),
                "scene_id": selection.scene_id,
                "seed": selection.seed,
                "source_height": source_size[1],
                "source_width": source_size[0],
                "task_document_path": _repo_path(doc_path),
                "task_document_sha256": _sha256_path(doc_path),
                "task_id": selection.task_id,
                "task_source_path": _repo_path(source_path),
                "task_source_sha256": _sha256_path(source_path),
                "task_versions": dict(sorted(generated.task_versions.items())),
                "width": size[0],
            }
        )

    hero_bytes, hero_size = _hero_montage(images_by_task)
    outputs[HERO_PATH] = hero_bytes
    hero_task_ids = [
        item.task_id
        for domain in ACTIVE_DOMAINS
        for item in GALLERY_SELECTIONS
        if item.domain == domain and item.hero
    ]
    manifest = {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "generator": {
            "path": "scripts/generate_release_gallery.py",
            "sha256": _sha256_path(Path(__file__).resolve()),
            "version": GALLERY_GENERATOR_VERSION,
        },
        "runtime": _runtime_provenance(),
        "provenance": {
            "catalog_path": _repo_path(CATALOG_PATH),
            "catalog_sha256": _sha256_path(CATALOG_PATH),
            "public_generation_source_root": "src/trace_tasks",
            "public_generation_source_sha256": _source_tree_sha256(),
            "revision_semantics": (
                "Exact release inputs are identified by content hashes; no pre-generation "
                "git revision is represented as containing these generated artifacts."
            ),
        },
        "summary": {
            "domain_count": len(ACTIVE_DOMAINS),
            "example_count": len(entries),
            "examples_per_domain": 2,
        },
        "image_policy": {
            "format": "PNG",
            "max_height": MAX_IMAGE_SIZE[1],
            "max_width": MAX_IMAGE_SIZE[0],
            "resampling": "Pillow.Image.Resampling.LANCZOS",
        },
        "hero": {
            "columns": HERO_COLUMNS,
            "height": hero_size[1],
            "image_path": _repo_path(HERO_PATH),
            "image_sha256": _sha256_bytes(hero_bytes),
            "task_ids": hero_task_ids,
            "width": hero_size[0],
        },
        "examples": entries,
    }
    return manifest, outputs


def check_gallery() -> list[str]:
    """Re-render in memory and byte-compare every generated gallery artifact."""

    problems: list[str] = []
    actual_manifest_bytes = (
        MANIFEST_PATH.read_bytes() if MANIFEST_PATH.is_file() else b""
    )
    try:
        committed_manifest = json.loads(actual_manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        committed_manifest = {}
        problems.append(f"invalid or missing: {_repo_path(MANIFEST_PATH)}")
    committed_runtime = committed_manifest.get("runtime")
    problems.extend(_validate_committed_runtime(committed_runtime))

    try:
        expected_manifest, expected_outputs = generate_gallery()
    except (GalleryError, OSError, ValueError) as exc:
        return problems + [f"gallery regeneration failed: {exc}"]
    if isinstance(committed_runtime, Mapping):
        # Runtime identifies the environment that produced the committed bytes.
        # The checker may be another supported Python minor version, so preserve
        # this provenance while byte-comparing all regenerated images.
        expected_manifest["runtime"] = dict(committed_runtime)

    expected_manifest_bytes = (
        json.dumps(expected_manifest, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    if actual_manifest_bytes != expected_manifest_bytes:
        problems.append(f"stale or missing: {_repo_path(MANIFEST_PATH)}")

    for path, expected in expected_outputs.items():
        actual = path.read_bytes() if path.is_file() else b""
        if actual != expected:
            problems.append(f"stale or missing: {_repo_path(path)}")

    expected_pngs = set(expected_outputs)
    if GALLERY_DIR.is_dir():
        unexpected = sorted(set(GALLERY_DIR.iterdir()) - expected_pngs)
        problems.extend(
            f"unexpected gallery path: {_repo_path(path)}" for path in unexpected
        )

    serialized = json.dumps(expected_manifest, ensure_ascii=False)
    forbidden_prefixes = (
        "/" + "home" + "/",
        "/" + "dev" + "/" + "shm",
        "/" + "workspace" + "/",
        "file" + "://",
    )
    for forbidden in forbidden_prefixes:
        if forbidden in serialized:
            problems.append(
                f"gallery manifest contains machine path prefix {forbidden!r}"
            )
    return problems


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic public Trace release gallery"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-render in memory and fail if any committed gallery artifact is stale",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        problems = check_gallery()
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            print(
                "gallery is stale; run `python scripts/generate_release_gallery.py`",
                file=sys.stderr,
            )
            return 1
        print("release gallery is up to date")
        return 0

    try:
        manifest, outputs = generate_gallery()
    except (GalleryError, OSError, ValueError) as exc:
        print(f"release gallery generation failed: {exc}", file=sys.stderr)
        return 1
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"wrote {_repo_path(path)}")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {_repo_path(MANIFEST_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
