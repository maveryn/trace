from __future__ import annotations

import json
from pathlib import Path
import re
import struct

import pytest

from scripts import check_docs_site

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_COMMIT_URL_RE = re.compile(
    r"(?:"
    r"https?://(?:www\.)?github\.com/maveryn/trace/"
    r"(?:blob|tree|commit)/[0-9a-fA-F]{7,40}"
    r"|"
    r"https?://raw\.githubusercontent\.com/maveryn/trace/"
    r"[0-9a-fA-F]{7,40}"
    r")(?=/|[?#]|$)"
)
ANSWER_ONLY_PROSE_RE = re.compile(r"\banswer(?:-|\s+)only\b", re.IGNORECASE)
RESEARCH_RESULTS_START = "<!-- trace-eval-v1-base-trace-table:start -->"
RESEARCH_RESULTS_END = "<!-- trace-eval-v1-base-trace-table:end -->"
RESULT_TABLE_PATHS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs/README.md",
    REPO_ROOT / "docs/research/README.md",
)
EXPECTED_RESEARCH_BENCHMARKS = (
    "ChartQAPro",
    "CharXivReason",
    "TableVQABench",
    "EvoChart",
    "MathVision",
    "MathVista",
    "MathVerse",
    "WeMath",
    "PhyX mini MC",
    "MMMU-ProVis",
    "RealWorldQA",
    "MMStar",
    "EmbSpatial",
    "SpatialVizBench COT",
    "CV-Bench 3D",
    "ERQA",
    "BLINK",
    "CountBenchQA",
    "CountQA",
    "TreeBench",
    "PuzzleVQA",
    "VisualPuzzles",
    "LogicVista",
    "MME-Reasoning",
)
PIPELINE_ASSET = REPO_ROOT / "docs/assets/paper-instance-pipeline/trace-instance-pipeline.png"
INFERENCE_EXAMPLE_ASSET = REPO_ROOT / "docs/assets/examples/trace-match3-validation-example.png"


def _metadata(canonical_url: str) -> str:
    image_url = check_docs_site.SOCIAL_IMAGE_URL
    return f"""
    <link rel="canonical" href="{canonical_url}">
    <meta property="og:type" content="website">
    <meta property="og:site_name" content="Trace">
    <meta property="og:title" content="Trace">
    <meta property="og:description" content="Visual reasoning documentation">
    <meta property="og:url" content="{canonical_url}">
    <meta property="og:image" content="{image_url}">
    <meta property="og:image:alt" content="Trace visual reasoning domains">
    <meta property="og:image:width" content="1512">
    <meta property="og:image:height" content="780">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="Trace">
    <meta name="twitter:description" content="Visual reasoning documentation">
    <meta name="twitter:image" content="{image_url}">
    <meta name="twitter:image:alt" content="Trace visual reasoning domains">
    """


def _png_header(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
    )


def _write_valid_site(site_root: Path) -> None:
    image = site_root / check_docs_site.SOCIAL_IMAGE_PATH
    image.parent.mkdir(parents=True)
    image.write_bytes(_png_header(*check_docs_site.SOCIAL_IMAGE_SIZE))
    stylesheet = site_root / "assets" / "styles" / "site.css"
    stylesheet.parent.mkdir(parents=True)
    stylesheet.write_text("body {}", encoding="utf-8")

    research = site_root / "research" / "index.html"
    research.parent.mkdir(parents=True)
    research.write_text(
        f"""<!doctype html>
        <html><head>{_metadata(check_docs_site.SITE_URL + 'research/')}</head>
        <body>
          <h1 id="methods">Research</h1>
          <a href="../?view=full#overview%20anchor">Home</a>
        </body></html>
        """,
        encoding="utf-8",
    )

    (site_root / "index.html").write_text(
        f"""<!doctype html>
        <html><head>{_metadata(check_docs_site.SITE_URL)}</head>
        <body>
          <h1 id="overview anchor">Trace</h1>
          <a href="#overview%20anchor">Overview</a>
          <a href="research/?source=home#methods">Research</a>
          <link rel="stylesheet" href="/trace/assets/styles/site.css?v=1">
          <img src="assets/paper%2Ddomain%2Dmontage/trace%2Dpaper%2Ddomain%2Dmontage.png">
        </body></html>
        """,
        encoding="utf-8",
    )


def test_check_docs_site_accepts_complete_local_site(tmp_path: Path) -> None:
    site_root = tmp_path / "site"
    site_root.mkdir()
    _write_valid_site(site_root)

    assert check_docs_site.check_docs_site(site_root) == 2


def test_check_docs_site_rejects_missing_target(tmp_path: Path) -> None:
    site_root = tmp_path / "site"
    site_root.mkdir()
    _write_valid_site(site_root)
    index = site_root / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "</body>", '<a href="missing/page/">Missing</a></body>'
        ),
        encoding="utf-8",
    )

    with pytest.raises(check_docs_site.DocsSiteError, match="target does not exist"):
        check_docs_site.check_docs_site(site_root)


def test_check_docs_site_rejects_missing_same_origin_target(tmp_path: Path) -> None:
    site_root = tmp_path / "site"
    site_root.mkdir()
    _write_valid_site(site_root)
    index = site_root / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "</body>",
            f'<a href="{check_docs_site.SITE_URL}missing/">Missing</a></body>',
        ),
        encoding="utf-8",
    )

    with pytest.raises(check_docs_site.DocsSiteError, match="target does not exist"):
        check_docs_site.check_docs_site(site_root)


def test_check_docs_site_rejects_missing_fragment(tmp_path: Path) -> None:
    site_root = tmp_path / "site"
    site_root.mkdir()
    _write_valid_site(site_root)
    index = site_root / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "research/?source=home#methods",
            "research/#missing-fragment",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        check_docs_site.DocsSiteError, match="fragment .* does not exist"
    ):
        check_docs_site.check_docs_site(site_root)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("/" + "home/jovyan/work/trace/image.png", "machine-specific path"),
        ("file://" + "/tmp/trace/image.png", "file URL"),
    ],
)
def test_check_docs_site_rejects_local_machine_urls(
    tmp_path: Path,
    url: str,
    message: str,
) -> None:
    site_root = tmp_path / "site"
    site_root.mkdir()
    _write_valid_site(site_root)
    index = site_root / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            "</body>", f'<img src="{url}"></body>'
        ),
        encoding="utf-8",
    )

    with pytest.raises(check_docs_site.DocsSiteError, match=message):
        check_docs_site.check_docs_site(site_root)


def test_check_docs_site_rejects_missing_social_metadata(tmp_path: Path) -> None:
    site_root = tmp_path / "site"
    site_root.mkdir()
    _write_valid_site(site_root)
    research = site_root / "research" / "index.html"
    research.write_text(
        research.read_text(encoding="utf-8").replace(
            '<meta name="twitter:image" '
            f'content="{check_docs_site.SOCIAL_IMAGE_URL}">',
            "",
        ),
        encoding="utf-8",
    )

    with pytest.raises(check_docs_site.DocsSiteError, match="twitter:image"):
        check_docs_site.check_docs_site(site_root)


def test_check_docs_site_rejects_wrong_social_image_dimensions(
    tmp_path: Path,
) -> None:
    site_root = tmp_path / "site"
    site_root.mkdir()
    _write_valid_site(site_root)
    image = site_root / check_docs_site.SOCIAL_IMAGE_PATH
    image.write_bytes(_png_header(1200, 630))

    with pytest.raises(check_docs_site.DocsSiteError, match="must be 1512 x 780"):
        check_docs_site.check_docs_site(site_root)


def test_public_markdown_uses_branch_stable_trace_urls() -> None:
    paths = [
        *REPO_ROOT.glob("*.md"),
        *(REPO_ROOT / "docs").rglob("*.md"),
    ]
    rlvr_root = REPO_ROOT / "rlvr"
    if rlvr_root.is_dir():
        paths.extend(rlvr_root.rglob("*.md"))

    violations: list[str] = []
    for path in sorted(set(paths)):
        for match in TRACE_COMMIT_URL_RE.finditer(path.read_text(encoding="utf-8")):
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            violations.append(f"{relative_path}: {match.group(0)}")

    assert not violations, (
        "self-links must use the named main, dev, or rlvr branch, not commit URLs:\n"
        + "\n".join(violations)
    )


def test_public_markdown_uses_neutral_output_mode_prose() -> None:
    paths = [*REPO_ROOT.glob("*.md")]
    for relative_root in (".agents", "docs", "examples", "rlvr", "scripts", "src"):
        root = REPO_ROOT / relative_root
        if root.is_dir():
            paths.extend(root.rglob("*.md"))

    violations: list[str] = []
    for path in sorted(set(paths)):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if ANSWER_ONLY_PROSE_RE.search(line):
                relative_path = path.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{relative_path}:{line_number}: {line.strip()}")

    assert not violations, (
        "use neutral public prose; keep answer_only only where it is a machine key:\n"
        + "\n".join(violations)
    )


def test_landing_readmes_stay_project_focused() -> None:
    forbidden = ("montage manifest", "provenance", "sha-256", "source hashes")
    for relative_path in ("README.md", "docs/README.md"):
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            assert phrase not in text, f"{relative_path} contains {phrase!r}"


def test_landing_pages_show_the_trace_instance_pipeline() -> None:
    assert PIPELINE_ASSET.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    references = {
        "README.md": "docs/assets/paper-instance-pipeline/trace-instance-pipeline.png",
        "docs/README.md": "assets/paper-instance-pipeline/trace-instance-pipeline.png",
    }
    for relative_path, reference in references.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert reference in text
        assert "How Trace Works" in text or "How Trace works" in text


def test_docs_homepage_uses_a_balanced_hero_wordmark() -> None:
    text = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert 'class="trace-hero__brand"' in text
    assert 'class="trace-hero__mark" src="assets/brand/trace-mark.svg"' in text


def test_readme_inference_uses_a_trace_validation_example() -> None:
    assert INFERENCE_EXAMPLE_ASSET.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/assets/examples/trace-match3-validation-example.png" in text
    assert '<img src="docs/assets/examples/trace-match3-validation-example.png"' not in text
    assert "Count the blue [#2D75E6] gems " in text
    assert '"in row 1.\\n"' in text


def _extract_base_trace_table(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.count(RESEARCH_RESULTS_START) == 1, path
    assert text.count(RESEARCH_RESULTS_END) == 1, path
    return text.split(RESEARCH_RESULTS_START, 1)[1].split(
        RESEARCH_RESULTS_END, 1
    )[0]


def test_research_page_has_all_base_trace_benchmark_results() -> None:
    research_path = REPO_ROOT / "docs" / "research" / "README.md"
    table = _extract_base_trace_table(research_path)
    for path in RESULT_TABLE_PATHS:
        assert _extract_base_trace_table(path) == table

    parsed_rows: list[list[str]] = []
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5:
            parsed_rows.append(cells)

    benchmark_rows = [
        row for row in parsed_rows if row[0] in EXPECTED_RESEARCH_BENCHMARKS
    ]
    assert tuple(row[0] for row in benchmark_rows) == EXPECTED_RESEARCH_BENCHMARKS

    score_pattern = re.compile(r"^\d+\.\d{2} ± \d+\.\d{2}$")
    trace_pattern = re.compile(r"^\d+\.\d{2} ± \d+\.\d{2} \([+-]\d+\.\d{2}\)$")
    for row in benchmark_rows:
        for index in (1, 3):
            assert score_pattern.fullmatch(row[index]), row
        for index in (2, 4):
            assert trace_pattern.fullmatch(row[index]), row

    results_path = REPO_ROOT / "rlvr" / "evaluation" / "trace_eval" / "results.json"
    suite_path = REPO_ROOT / "rlvr" / "evaluation" / "trace_eval" / "suite.v1.json"
    if not results_path.is_file() or not suite_path.is_file():
        return

    results = json.loads(results_path.read_text(encoding="utf-8"))
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    assert tuple(item["display"] for item in suite["benchmarks"]) == (
        EXPECTED_RESEARCH_BENCHMARKS
    )

    summaries = {
        (row["model_id"], row["benchmark_id"]): row
        for row in results["scores"]["benchmark_summaries"]
    }
    deltas = {
        (comparison["model_id"], row["benchmark_id"]): row
        for comparison in results["comparisons"]
        for row in comparison["benchmark_deltas"]
    }
    model_ids = (
        "qwen2.5-vl-3b-base",
        "trace-qwen2.5-vl-3b",
        "qwen2.5-vl-7b-base",
        "trace-qwen2.5-vl-7b",
    )

    def format_summary(row: dict[str, object]) -> str:
        return f"{float(row['mean']):.2f} ± {float(row['stddev']):.2f}"

    def format_trace(summary: dict[str, object], delta: dict[str, object]) -> str:
        return f"{format_summary(summary)} ({float(delta['mean']):+.2f})"

    rows_by_label = {row[0]: row[1:] for row in benchmark_rows}
    for benchmark in suite["benchmarks"]:
        benchmark_id = benchmark["key"]
        expected = [
            format_summary(summaries[(model_ids[0], benchmark_id)]),
            format_trace(
                summaries[(model_ids[1], benchmark_id)],
                deltas[(model_ids[1], benchmark_id)],
            ),
            format_summary(summaries[(model_ids[2], benchmark_id)]),
            format_trace(
                summaries[(model_ids[3], benchmark_id)],
                deltas[(model_ids[3], benchmark_id)],
            ),
        ]
        assert rows_by_label[benchmark["display"]] == expected
