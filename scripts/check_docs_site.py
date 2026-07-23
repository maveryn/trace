#!/usr/bin/env python3
"""Validate local links and social metadata in the generated documentation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from urllib.parse import unquote, urlsplit

SITE_URL = "https://maveryn.github.io/trace/"
SITE_PATH_PREFIX = "/trace"
SOCIAL_IMAGE_URL = (
    f"{SITE_URL}assets/paper-domain-montage/trace-paper-domain-montage.png"
)
SOCIAL_IMAGE_PATH = Path("assets/paper-domain-montage/trace-paper-domain-montage.png")
SOCIAL_IMAGE_SIZE = (1512, 780)
REQUIRED_PAGES = {Path("index.html"), Path("research/index.html")}
NONCANONICAL_PAGES = {Path("404.html")}
_SITE_PARTS = urlsplit(SITE_URL)
_ROOT = "/"
MACHINE_PATH_PREFIXES = tuple(
    f"{_ROOT}{relative}/"
    for relative in (
        "home",
        "Users",
        "root",
        "tmp",
        "var/tmp",
        "mnt",
        "opt",
        "scratch",
        "workspace",
        "dev/shm",
    )
)
WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


class DocsSiteError(RuntimeError):
    """Raised when the generated documentation violates the site contract."""


@dataclass
class ParsedPage:
    references: list[tuple[str, str]] = field(default_factory=list)
    fragments: set[str] = field(default_factory=set)
    canonical_urls: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page = ParsedPage()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = {name.lower(): value or "" for name, value in attrs}
        for attribute in ("href", "src"):
            if attribute in values:
                self.page.references.append((attribute, values[attribute]))

        fragment = values.get("id")
        if fragment:
            self.page.fragments.add(fragment)
        if tag.lower() == "a" and values.get("name"):
            self.page.fragments.add(values["name"])

        if tag.lower() == "link":
            relations = values.get("rel", "").lower().split()
            if "canonical" in relations and values.get("href"):
                self.page.canonical_urls.append(values["href"])

        if tag.lower() == "meta":
            key = values.get("property") or values.get("name")
            if key:
                self.page.metadata[key.lower()] = values.get("content", "")


def _parse_page(path: Path) -> ParsedPage:
    parser = _PageParser()
    try:
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
    except (OSError, UnicodeError) as exc:
        raise DocsSiteError(f"could not parse {path}: {exc}") from exc
    return parser.page


def _looks_like_machine_path(value: str) -> bool:
    unquoted = unquote(value)
    if unquoted.startswith("\\\\"):
        return True
    decoded = unquoted.replace("\\", "/")
    return (
        decoded.startswith("~/")
        or decoded.startswith("//./")
        or decoded.startswith("//?/")
        or bool(WINDOWS_PATH_RE.match(decoded))
        or any(decoded.startswith(prefix) for prefix in MACHINE_PATH_PREFIXES)
    )


def _relative_display(path: Path, site_root: Path) -> str:
    return path.relative_to(site_root).as_posix()


def _expected_canonical_url(path: Path) -> str:
    relative = path.as_posix()
    if path.name == "index.html":
        parent = path.parent.as_posix()
        suffix = "" if parent == "." else f"{parent}/"
    else:
        suffix = relative
    return f"{SITE_URL}{suffix}"


def _read_png_size(path: Path) -> tuple[int, int]:
    try:
        header = path.read_bytes()[:24]
    except OSError as exc:
        raise DocsSiteError(
            f"could not read social preview image {path}: {exc}"
        ) from exc
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise DocsSiteError(f"social preview image is not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def _resolve_local_target(
    site_root: Path,
    source: Path,
    raw_url: str,
) -> tuple[Path | None, str | None]:
    """Return the resolved local path and decoded fragment for an HTML URL."""

    stripped = raw_url.strip()
    if _looks_like_machine_path(stripped):
        raise DocsSiteError(f"machine-specific path is not allowed: {raw_url!r}")

    parsed = urlsplit(stripped)
    if parsed.scheme.lower() == "file":
        raise DocsSiteError(f"file URL is not allowed: {raw_url!r}")
    if parsed.netloc:
        same_origin = parsed.netloc.lower() == _SITE_PARTS.netloc.lower() and (
            not parsed.scheme or parsed.scheme.lower() == _SITE_PARTS.scheme
        )
        if not same_origin:
            return None, None
    elif parsed.scheme:
        return None, None

    decoded_path = unquote(parsed.path).replace("\\", "/")
    if _looks_like_machine_path(decoded_path):
        raise DocsSiteError(f"machine-specific path is not allowed: {raw_url!r}")

    if decoded_path.startswith("/"):
        if decoded_path == SITE_PATH_PREFIX:
            decoded_path = "/"
        elif decoded_path.startswith(f"{SITE_PATH_PREFIX}/"):
            decoded_path = decoded_path[len(SITE_PATH_PREFIX) :]
        unresolved = site_root.joinpath(decoded_path.lstrip("/"))
    elif decoded_path:
        unresolved = source.parent.joinpath(decoded_path)
    else:
        unresolved = source

    target = unresolved.resolve()
    try:
        target.relative_to(site_root)
    except ValueError as exc:
        raise DocsSiteError(f"path escapes the generated site: {raw_url!r}") from exc

    if target.is_dir() or decoded_path.endswith("/"):
        target = target / "index.html"
    elif not target.exists() and not PurePosixPath(decoded_path).suffix:
        index_target = target / "index.html"
        if index_target.exists():
            target = index_target

    return target, unquote(parsed.fragment) or None


def _check_metadata(
    site_root: Path,
    pages: dict[Path, ParsedPage],
    errors: list[str],
) -> None:
    image_path = site_root / SOCIAL_IMAGE_PATH
    if not image_path.is_file():
        errors.append(f"missing social preview image: {SOCIAL_IMAGE_PATH.as_posix()}")
    else:
        try:
            observed_size = _read_png_size(image_path)
        except DocsSiteError as exc:
            errors.append(str(exc))
        else:
            if observed_size != SOCIAL_IMAGE_SIZE:
                errors.append(
                    "social preview image must be "
                    f"{SOCIAL_IMAGE_SIZE[0]} x {SOCIAL_IMAGE_SIZE[1]}; "
                    f"observed {observed_size[0]} x {observed_size[1]}"
                )

    required_exact = {
        "og:type": "website",
        "og:site_name": "Trace",
        "og:image": SOCIAL_IMAGE_URL,
        "og:image:width": str(SOCIAL_IMAGE_SIZE[0]),
        "og:image:height": str(SOCIAL_IMAGE_SIZE[1]),
        "twitter:card": "summary_large_image",
        "twitter:image": SOCIAL_IMAGE_URL,
    }
    required_nonempty = (
        "og:title",
        "og:description",
        "og:image:alt",
        "twitter:title",
        "twitter:description",
        "twitter:image:alt",
    )

    for required_page in sorted(REQUIRED_PAGES):
        if required_page not in pages:
            errors.append(
                f"required metadata page is missing: {required_page.as_posix()}"
            )

    for page_path, page in pages.items():
        expected_metadata = required_exact
        if page_path in NONCANONICAL_PAGES:
            if page.canonical_urls:
                errors.append(
                    f"{page_path.as_posix()}: noncanonical page must not declare "
                    f"a canonical URL; observed {page.canonical_urls!r}"
                )
            if "og:url" in page.metadata:
                errors.append(
                    f"{page_path.as_posix()}: noncanonical page must not declare og:url"
                )
        else:
            expected_url = _expected_canonical_url(page_path)
            if page.canonical_urls != [expected_url]:
                errors.append(
                    f"{page_path.as_posix()}: canonical URL must be {expected_url!r}; "
                    f"observed {page.canonical_urls!r}"
                )
            expected_metadata = {**required_exact, "og:url": expected_url}

        for key, expected in expected_metadata.items():
            observed = page.metadata.get(key)
            if observed != expected:
                errors.append(
                    f"{page_path.as_posix()}: {key} must be {expected!r}; "
                    f"observed {observed!r}"
                )
        for key in required_nonempty:
            if not page.metadata.get(key, "").strip():
                errors.append(
                    f"{page_path.as_posix()}: required metadata {key} is missing or empty"
                )


def check_docs_site(site_dir: str | Path) -> int:
    """Validate a generated MkDocs tree and return the number of HTML pages."""

    site_root = Path(site_dir).resolve()
    if not site_root.is_dir():
        raise DocsSiteError(f"generated site directory does not exist: {site_root}")

    html_paths = sorted(site_root.rglob("*.html"))
    if not html_paths:
        raise DocsSiteError(f"generated site contains no HTML files: {site_root}")

    parsed_by_absolute = {path: _parse_page(path) for path in html_paths}
    parsed_by_relative = {
        path.relative_to(site_root): page for path, page in parsed_by_absolute.items()
    }
    errors: list[str] = []

    for source, page in parsed_by_absolute.items():
        source_name = _relative_display(source, site_root)
        for attribute, raw_url in page.references:
            try:
                target, fragment = _resolve_local_target(
                    site_root,
                    source,
                    raw_url,
                )
            except DocsSiteError as exc:
                errors.append(f"{source_name}: {attribute} {exc}")
                continue

            if target is None:
                continue
            if not target.is_file():
                errors.append(
                    f"{source_name}: {attribute} target does not exist: {raw_url!r}"
                )
                continue
            if fragment:
                target_page = parsed_by_absolute.get(target)
                if target_page is None:
                    errors.append(
                        f"{source_name}: fragment targets a non-HTML file: {raw_url!r}"
                    )
                elif fragment not in target_page.fragments:
                    errors.append(
                        f"{source_name}: fragment {fragment!r} does not exist in "
                        f"{_relative_display(target, site_root)}"
                    )

    _check_metadata(site_root, parsed_by_relative, errors)
    if errors:
        details = "\n".join(f"- {error}" for error in sorted(set(errors)))
        raise DocsSiteError(f"documentation site validation failed:\n{details}")
    return len(html_paths)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated documentation links and metadata."
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path("site"),
        help="generated MkDocs directory (default: site)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        page_count = check_docs_site(args.site_dir)
    except DocsSiteError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"documentation site validation passed: {page_count} HTML files checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
