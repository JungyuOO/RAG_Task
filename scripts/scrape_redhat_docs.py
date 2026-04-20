"""Scrape selected docs.redhat.com OCP books into clean markdown using Playwright.

This script prefers `full-html/<slug>/index` and falls back to chapter-wise
`html/<slug>` crawling when the full-html page is unavailable or too small.

Usage:
    python scripts/scrape_redhat_docs.py --version 4.20 --locale en --slug-file scripts/allowlists/redhat_ocp_420_curated.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from playwright.async_api import Browser, Page, async_playwright

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "corpus" / "pdfs" / "official" / "en"
DEFAULT_BACKUP_ROOT = ROOT / "data" / "corpus" / "pdfs" / "legacy" / "official-en-pre-redhat-html"
INVENTORY_JSON = ROOT / "tests" / "results" / "redhat-docs-inventory" / "ocp-4.20-en.json"


class RedHatMarkdownConverter(MarkdownConverter):
    def convert_blockquote(self, el, text, convert_as_inline=None, **kwargs):  # type: ignore[override]
        del convert_as_inline, kwargs
        body = text.strip("\n")
        if not body:
            return "\n"
        lines = [f"> {line}" if line.strip() else ">" for line in body.splitlines()]
        return "\n" + "\n".join(lines) + "\n\n"

    def convert_pre(self, el, text=None, convert_as_inline=None, **kwargs):  # type: ignore[override]
        del kwargs, convert_as_inline, text
        code = el.find("code")
        raw = code.get_text("", strip=False) if code else el.get_text("", strip=False)
        raw = raw.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
        raw = "\n".join(line.rstrip() for line in raw.splitlines()).strip("\n")
        classes = []
        if code is not None:
            classes.extend(code.get("class", []))
        classes.extend(el.get("class", []))
        class_text = " ".join(str(item) for item in classes)
        match = re.search(r"language-([a-zA-Z0-9_+-]+)", class_text)
        lang = match.group(1) if match else ""
        return f"\n```{lang}\n{raw}\n```\n\n"


@dataclass(slots=True)
class ScrapeResult:
    slug: str
    title: str
    mode: str
    markdown_path: str
    source_url: str
    chapter_count: int
    article_count: int
    status: str
    error: str = ""


def landing_url(*, version: str, locale: str, slug: str) -> str:
    return f"https://docs.redhat.com/{locale}/documentation/openshift_container_platform/{version}/html/{slug}"


def full_html_url(*, version: str, locale: str, slug: str) -> str:
    return f"https://docs.redhat.com/{locale}/documentation/openshift_container_platform/{version}/full-html/{slug}/index"


def _clean_non_code_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]*\n\s*\n", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def clean_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned_blocks: list[str] = []
    buffer: list[str] = []
    in_code_fence = False

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_fence:
                buffer.append(line.rstrip())
                cleaned_blocks.append("\n".join(buffer).strip("\n"))
                buffer = []
                in_code_fence = False
            else:
                if buffer:
                    cleaned_blocks.append(_clean_non_code_whitespace("\n".join(buffer)).strip("\n"))
                    buffer = []
                buffer.append(line.rstrip())
                in_code_fence = True
            continue
        buffer.append(line if in_code_fence else line)

    if buffer:
        if in_code_fence:
            cleaned_blocks.append("\n".join(buffer).strip("\n"))
        else:
            cleaned_blocks.append(_clean_non_code_whitespace("\n".join(buffer)).strip("\n"))

    text = "\n\n".join(block for block in cleaned_blocks if block)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def strip_trailing_reference_sections(text: str) -> str:
    value = str(text or "")
    cut_patterns = [
        re.compile(r"(?im)^\*\*Additional resources\*\*\s*$"),
        re.compile(r"(?im)^##\s+Legal Notice\s*$"),
    ]

    cut_index: int | None = None
    for pattern in cut_patterns:
        match = pattern.search(value)
        if match is None:
            continue
        cut_index = match.start() if cut_index is None else min(cut_index, match.start())

    if cut_index is not None:
        value = value[:cut_index]

    value = re.sub(
        r"(?is)\n*\[Previous\]\([^)]+\)\[Next\]\([^)]+\)\s*$",
        "",
        value,
    )
    return value.rstrip() + "\n"


def strip_additional_resources_sections(text: str) -> str:
    lines = str(text or "").splitlines()
    kept: list[str] = []
    skipping = False
    pending_links: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_heading = bool(re.match(r"^#{2,6}\s+", stripped))
        is_additional_resources = bool(
            re.match(r"^(?:#{2,6}\s+)?(?:\d+(?:\.\d+)*\.?\s+)?Additional resources\s*$", stripped, re.IGNORECASE)
            or re.match(r"^>\s*Additional resources\s*$", stripped, re.IGNORECASE)
            or re.match(r"^\*\*Additional resources\*\*\s*$", stripped, re.IGNORECASE)
        )
        is_resource_link = bool(re.match(r"^-\s+\[.+\]\(https://docs\.redhat\.com/.+\)\s*$", stripped, re.IGNORECASE))

        if is_additional_resources:
            skipping = True
            pending_links = []
            continue

        if skipping and is_heading:
            skipping = False
            pending_links = []

        if skipping:
            continue

        if is_resource_link:
            pending_links.append(line)
            continue

        if pending_links:
            if not (len(pending_links) >= 2 and is_heading):
                kept.extend(pending_links)
            pending_links = []

        kept.append(line)

    if pending_links and len(pending_links) < 2:
        kept.extend(pending_links)

    return "\n".join(kept).rstrip() + "\n"


def strip_additional_resources_mentions(text: str) -> str:
    lines = str(text or "").splitlines()
    cleaned_lines: list[str] = []
    patterns = [
        re.compile(r"^.*Additional resources section.*$", re.IGNORECASE),
        re.compile(r"^.*following \"Additional resources\" section.*$", re.IGNORECASE),
        re.compile(r"^.*links in the Additional resources:.*$", re.IGNORECASE),
        re.compile(r"^.*For more information on these tasks, see the \*Additional resources\* section\..*$", re.IGNORECASE),
        re.compile(r"^.*For more information, see .*Additional resources.*$", re.IGNORECASE),
    ]

    for line in lines:
        if any(pattern.search(line.strip()) for pattern in patterns):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).rstrip() + "\n"


def transform_custom_elements(article_html: str) -> str:
    soup = BeautifulSoup(article_html, "html.parser")

    for tag in soup.select("rh-tooltip, .copy-link-tooltip"):
        tag.decompose()

    titlepage = soup.select_one(".titlepage")
    if titlepage is not None:
        titlepage.decompose()

    for heading in soup.select("h2, h3, h4, h5, h6"):
        anchor = heading.select_one("a.anchor-heading")
        if anchor is not None:
            heading.clear()
            heading.append(anchor.get_text(" ", strip=True))

    for code_block in list(soup.select("rh-code-block")):
        pre = code_block.find("pre")
        if pre is not None:
            code_block.replace_with(pre)
            continue
        code = code_block.find("code")
        if code is not None:
            pre = soup.new_tag("pre")
            pre.append(code.extract())
            code_block.replace_with(pre)

    for alert in list(soup.select("rh-alert")):
        state = str(alert.get("state") or "").strip().title() or "Note"
        body = alert.get_text("\n", strip=True)
        quote = soup.new_tag("blockquote")
        para = soup.new_tag("p")
        para.string = f"**{state}:** {body}"
        quote.append(para)
        alert.replace_with(quote)

    for hidden in soup.select("[hidden], script, style"):
        hidden.decompose()

    return str(soup)


def html_to_markdown(article_html: str) -> str:
    cleaned_html = transform_custom_elements(article_html)
    markdown = RedHatMarkdownConverter(heading_style="ATX", bullets="-").convert(cleaned_html)
    markdown = re.sub(r"^(#{2,6})\s+\[(.*?)\]\((#[^)]+)\)\s*$", r"\1 \2", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"(?m)^(Expand|Show more)\s*$\n?", "", markdown)
    markdown = clean_whitespace(markdown)
    markdown = strip_additional_resources_sections(markdown)
    markdown = strip_additional_resources_mentions(markdown)
    markdown = strip_trailing_reference_sections(markdown)
    markdown = re.sub(r"\n```(\w*)\n\s*\n", r"\n```\1\n", markdown)
    return markdown


async def goto(page: Page, url: str) -> None:
    await page.goto(url, wait_until="domcontentloaded", timeout=120000)
    await page.wait_for_timeout(5000)


async def get_article_html(page: Page) -> str | None:
    article = await page.query_selector("main#main-content article")
    if article is None:
        return None
    return await article.evaluate("(el) => el.outerHTML")


async def collect_chapter_links(page: Page, html_url: str) -> list[str]:
    links = await page.eval_on_selector_all(
        "main#main-content a[href]",
        """
        (els) => els
          .map((a) => ({ href: a.href, text: (a.innerText || a.textContent || '').trim() }))
          .filter((item) => item.href && item.text)
        """,
    )
    current_path = urlparse(html_url).path.rstrip("/")
    chapter_links: list[str] = []
    for item in links:
        href = str(item.get("href") or "")
        href_path = urlparse(href).path.rstrip("/")
        if current_path not in href_path:
            continue
        if href_path == current_path:
            continue
        if href not in chapter_links:
            chapter_links.append(href)
    return chapter_links


def read_slug_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def read_inventory(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["slug"]): item for item in items}


def backup_existing(target: Path, backup_root: Path) -> None:
    if not target.exists():
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / target.name
    if backup_path.exists():
        backup_path.unlink()
    shutil.move(str(target), str(backup_path))


async def scrape_one(browser: Browser, *, version: str, locale: str, slug: str, output_root: Path, backup_root: Path | None) -> ScrapeResult:
    page = await browser.new_page()
    try:
        target = output_root / f"{slug}.md"
        book_title = slug.replace("_", " ").title()

        full_url = full_html_url(version=version, locale=locale, slug=slug)
        await goto(page, full_url)
        article_html = await get_article_html(page)
        if article_html is not None:
            markdown = html_to_markdown(article_html)
            if len(markdown) > 400:
                if backup_root is not None:
                    backup_existing(target, backup_root)
                output_root.mkdir(parents=True, exist_ok=True)
                target.write_text(markdown, encoding="utf-8")
                return ScrapeResult(
                    slug=slug,
                    title=book_title,
                    mode="full-html",
                    markdown_path=str(target),
                    source_url=full_url,
                    chapter_count=0,
                    article_count=1,
                    status="ok",
                )

        html_url = landing_url(version=version, locale=locale, slug=slug)
        await goto(page, html_url)
        chapter_links = await collect_chapter_links(page, html_url)
        chapter_markdowns: list[str] = []
        for chapter_url in chapter_links:
            await goto(page, chapter_url)
            chapter_article = await get_article_html(page)
            if chapter_article is None:
                continue
            chapter_markdowns.append(html_to_markdown(chapter_article))

        if not chapter_markdowns:
            return ScrapeResult(
                slug=slug,
                title=book_title,
                mode="html",
                markdown_path=str(target),
                source_url=html_url,
                chapter_count=len(chapter_links),
                article_count=0,
                status="failed",
                error="No article content could be extracted from html fallback.",
            )

        combined = clean_whitespace("\n\n".join(part.strip() for part in chapter_markdowns if part.strip()))
        if backup_root is not None:
            backup_existing(target, backup_root)
        output_root.mkdir(parents=True, exist_ok=True)
        target.write_text(combined, encoding="utf-8")
        return ScrapeResult(
            slug=slug,
            title=book_title,
            mode="html-chapters",
            markdown_path=str(target),
            source_url=html_url,
            chapter_count=len(chapter_links),
            article_count=len(chapter_markdowns),
            status="ok",
        )
    except Exception as exc:
        return ScrapeResult(
            slug=slug,
            title=slug.replace("_", " ").title(),
            mode="unknown",
            markdown_path=str(output_root / f"{slug}.md"),
            source_url=full_html_url(version=version, locale=locale, slug=slug),
            chapter_count=0,
            article_count=0,
            status="failed",
            error=str(exc),
        )
    finally:
        await page.close()


async def scrape_books(*, version: str, locale: str, slugs: list[str], output_root: Path, backup_root: Path | None) -> list[ScrapeResult]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            semaphore = asyncio.Semaphore(3)

            async def worker(slug: str) -> ScrapeResult:
                async with semaphore:
                    return await scrape_one(
                        browser,
                        version=version,
                        locale=locale,
                        slug=slug,
                        output_root=output_root,
                        backup_root=backup_root,
                    )

            return await asyncio.gather(*(worker(slug) for slug in slugs))
        finally:
            await browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape selected Red Hat OCP docs into clean markdown.")
    parser.add_argument("--version", default="4.20")
    parser.add_argument("--locale", default="en")
    parser.add_argument("--slug-file", default=str(ROOT / "scripts" / "allowlists" / "redhat_ocp_420_curated.txt"))
    parser.add_argument("--slug", action="append", default=[])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUP_ROOT))
    parser.add_argument("--inventory-json", default=str(INVENTORY_JSON))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slugs = [slug.strip() for slug in args.slug if slug.strip()]
    if not slugs:
        slugs = read_slug_file(Path(args.slug_file))

    inventory = read_inventory(Path(args.inventory_json))
    selected = []
    for slug in slugs:
        record = inventory.get(slug, {})
        selected.append(
            {
                "slug": slug,
                "title": record.get("title", slug.replace("_", " ").title()),
                "full_html_supported": record.get("full_html_supported"),
                "html_chapter_count": record.get("html_chapter_count"),
                "matched_topics": record.get("matched_topics", []),
            }
        )

    if args.dry_run:
        print(json.dumps({"count": len(selected), "selected": selected}, ensure_ascii=False, indent=2))
        return

    results = asyncio.run(
        scrape_books(
            version=args.version,
            locale=args.locale,
            slugs=slugs,
            output_root=Path(args.output_root),
            backup_root=Path(args.backup_root) if args.backup_root else None,
        )
    )
    payload = {
        "count": len(results),
        "ok": sum(1 for item in results if item.status == "ok"),
        "failed": sum(1 for item in results if item.status != "ok"),
        "results": [asdict(item) for item in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
