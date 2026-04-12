"""Bulk import operations-focused official html-single OCP docs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from scripts.import_ocp_html_single import import_ocp_html_single

OPS_HTML_SINGLE_SLUGS = [
    "advanced_networking",
    "authentication_and_authorization",
    "backup_and_restore",
    "cli_tools",
    "images",
    "ingress_and_load_balancing",
    "logging",
    "machine_configuration",
    "machine_management",
    "networking_overview",
    "nodes",
    "observability_overview",
    "postinstallation_configuration",
    "registry",
    "security_and_compliance",
    "storage",
    "support",
    "updating_clusters",
    "validation_and_troubleshooting",
]


def build_html_single_url(*, version: str, locale: str, slug: str) -> str:
    return (
        f"https://docs.redhat.com/{locale}/documentation/openshift_container_platform/"
        f"{version}/html-single/{slug}/index"
    )


def fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (RAG-Task html-single importer)"})
    with urlopen(request, timeout=120) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="ignore")


def import_many(*, version: str, locale: str, slugs: list[str]) -> list[dict]:
    settings = get_settings()
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ocp-html-single-import-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for slug in slugs:
            url = build_html_single_url(version=version, locale=locale, slug=slug)
            html_path = tmp_root / f"{slug}.html"
            meta_path = tmp_root / f"{slug}.meta.json"
            html_path.write_text(fetch_html(url), encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    {
                        "book_slug": slug,
                        "book_title": slug.replace("_", " ").title(),
                        "ocp_version": version,
                        "docs_language": locale,
                        "resolved_language": locale,
                        "source_url": url,
                        "resolved_source_url": url,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = import_ocp_html_single(html_path, meta_path, settings.rag_source_dir, settings.rag_extract_dir)
            results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="4.20")
    parser.add_argument("--locale", default="en")
    parser.add_argument("--slug", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slugs = [slug.strip() for slug in args.slug if slug.strip()] or OPS_HTML_SINGLE_SLUGS
    results = import_many(version=args.version, locale=args.locale, slugs=slugs)
    print(json.dumps({"count": len(results), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
