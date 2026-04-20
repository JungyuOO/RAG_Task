"""Inventory docs.redhat.com OCP books and verify full-html support with Playwright.

Usage:
    python scripts/inventory_redhat_docs.py --version 4.20 --locale en
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page, async_playwright

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "tests" / "results" / "redhat-docs-inventory"
BOOK_URL_RE = re.compile(
    r"/documentation/openshift_container_platform/(?P<version>[^/]+)/html/(?P<slug>[^/#?]+)"
)

TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "nodes": ("nodes", "machine", "scalability"),
    "auth_security": ("authentication", "authorization", "security", "compliance", "network_security"),
    "storage_backup": ("storage", "backup", "etcd"),
    "images_registry_builds": ("images", "registry", "building_applications", "builds_"),
    "networking": (
        "networking",
        "network_",
        "ingress",
        "load_balancing",
        "multiple_networks",
        "ovn-kubernetes",
        "kubernetes_nmstate",
        "advanced_networking",
    ),
    "cicd_gitops": ("cicd", "gitops", "pipelines", "shipwright"),
    "cluster_ops": ("postinstallation", "updating_clusters", "validation_and_troubleshooting", "cli_tools", "support"),
}


@dataclass(slots=True)
class BookRecord:
    slug: str
    title: str
    category: str
    html_url: str
    full_html_url: str
    full_html_supported: bool = False
    full_html_article_text_len: int = 0
    full_html_heading_count: int = 0
    html_chapter_count: int = 0
    html_chapter_samples: list[str] | None = None
    matched_topics: list[str] | None = None


def landing_url(*, version: str, locale: str) -> str:
    return f"https://docs.redhat.com/{locale}/documentation/openshift_container_platform/{version}"


def build_html_url(*, version: str, locale: str, slug: str) -> str:
    return f"https://docs.redhat.com/{locale}/documentation/openshift_container_platform/{version}/html/{slug}"


def build_full_html_url(*, version: str, locale: str, slug: str) -> str:
    return f"https://docs.redhat.com/{locale}/documentation/openshift_container_platform/{version}/full-html/{slug}/index"


async def _goto(page: Page, url: str) -> None:
    await page.goto(url, wait_until="domcontentloaded", timeout=120000)
    await page.wait_for_timeout(5000)


def extract_books_from_html(html: str, *, version: str, locale: str) -> list[BookRecord]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("main#main-content")
    if main is None:
        return []

    records: dict[str, BookRecord] = {}
    current_category = ""
    for node in main.descendants:
        name = getattr(node, "name", None)
        if name in {"h2", "h3"}:
            current_category = " ".join(node.get_text(" ", strip=True).split())
            continue
        if name != "a":
            continue
        href = str(node.get("href") or "").strip()
        if not href:
            continue
        match = BOOK_URL_RE.search(href)
        if not match or match.group("version") != version:
            continue
        slug = match.group("slug")
        title = " ".join(node.get_text(" ", strip=True).split())
        if not title:
            continue
        if slug in records:
            continue
        records[slug] = BookRecord(
            slug=slug,
            title=title,
            category=current_category,
            html_url=build_html_url(version=version, locale=locale, slug=slug),
            full_html_url=build_full_html_url(version=version, locale=locale, slug=slug),
        )
    return list(records.values())


def classify_topics(slug: str, title: str) -> list[str]:
    haystack = f"{slug} {title}".casefold()
    matches = [
        topic
        for topic, patterns in TOPIC_PATTERNS.items()
        if any(pattern.casefold() in haystack for pattern in patterns)
    ]
    return matches


async def inspect_book(browser: Browser, record: BookRecord) -> BookRecord:
    page = await browser.new_page()
    try:
        await _goto(page, record.full_html_url)
        article = await page.query_selector("main#main-content article")
        if article is not None:
            text = ((await article.inner_text()) or "").strip()
            headings = await article.query_selector_all("h1, h2, h3, h4")
            record.full_html_article_text_len = len(text)
            record.full_html_heading_count = len(headings)
            record.full_html_supported = len(text) >= 400 and len(headings) >= 4

        await _goto(page, record.html_url)
        links = await page.eval_on_selector_all(
            "main#main-content a[href]",
            """
            (els) => els
              .map((a) => ({
                href: a.href,
                text: (a.innerText || a.textContent || '').trim(),
              }))
              .filter((item) => item.href && item.text)
            """,
        )
        chapter_links: list[str] = []
        parsed_html_path = urlparse(record.html_url).path.rstrip("/")
        for item in links:
            href = str(item.get("href") or "")
            if parsed_html_path not in href:
                continue
            href_path = urlparse(href).path.rstrip("/")
            if href_path == parsed_html_path:
                continue
            if href not in chapter_links:
                chapter_links.append(href)
        record.html_chapter_count = len(chapter_links)
        record.html_chapter_samples = chapter_links[:5]
        record.matched_topics = classify_topics(record.slug, record.title)
        return record
    finally:
        await page.close()


async def inventory_books(*, version: str, locale: str) -> list[BookRecord]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            landing = await browser.new_page()
            await _goto(landing, landing_url(version=version, locale=locale))
            html = await landing.content()
            await landing.close()

            books = extract_books_from_html(html, version=version, locale=locale)
            semaphore = asyncio.Semaphore(4)

            async def worker(item: BookRecord) -> BookRecord:
                async with semaphore:
                    return await inspect_book(browser, item)

            return await asyncio.gather(*(worker(item) for item in books))
        finally:
            await browser.close()


def render_markdown(records: Iterable[BookRecord], *, version: str, locale: str) -> str:
    lines = [
        f"# Red Hat Docs Inventory",
        "",
        f"- version: `{version}`",
        f"- locale: `{locale}`",
        "",
        "| title | slug | category | full-html | chapters | topics |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for item in records:
        lines.append(
            "| {title} | `{slug}` | {category} | {full_html} | {chapter_count} | {topics} |".format(
                title=item.title.replace("|", "\\|"),
                slug=item.slug,
                category=(item.category or "-").replace("|", "\\|"),
                full_html="yes" if item.full_html_supported else "no",
                chapter_count=item.html_chapter_count,
                topics=", ".join(item.matched_topics or []) or "-",
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory OCP books from docs.redhat.com with Playwright.")
    parser.add_argument("--version", default="4.20")
    parser.add_argument("--locale", default="en")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = asyncio.run(inventory_books(version=args.version, locale=args.locale))
    records = sorted(records, key=lambda item: (item.category.casefold(), item.title.casefold(), item.slug))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else RESULTS_DIR / f"ocp-{args.version}-{args.locale}.json"
    output_md = Path(args.output_md) if args.output_md else RESULTS_DIR / f"ocp-{args.version}-{args.locale}.md"

    output_json.write_text(json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(records, version=args.version, locale=args.locale), encoding="utf-8")

    summary = {
        "count": len(records),
        "full_html_supported": sum(1 for item in records if item.full_html_supported),
        "html_only": sum(1 for item in records if not item.full_html_supported),
        "output_json": str(output_json),
        "output_md": str(output_md),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
