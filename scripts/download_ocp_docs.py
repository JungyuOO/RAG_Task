"""Download OpenShift Container Platform PDFs from docs.redhat.com.

This script is designed for RAG corpus curation, not for mirroring every
available Red Hat document. It supports:

- friendly category aliases such as ``networking`` or ``lightspeed``
- version-aware category discovery from the official docs site
- PDF URL resolution from real documentation pages
- version-scoped output folders under ``data/corpus/pdfs/ocp-<version>/``
- per-version and global manifest generation
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx

BASE_URL = "https://docs.redhat.com/en/documentation/openshift_container_platform"
PRODUCT_NAME = "OpenShift_Container_Platform"
USER_AGENT = "Mozilla/5.0 (OCP-Doc-Downloader/2.0)"
RETRY_DELAYS = [2, 5, 10]

DOC_LINK_RE = re.compile(
    r'href="(?P<href>/en/documentation/openshift_container_platform/'
    r'(?P<version>[^"/?#]+)/(?P<kind>html|html-single|pdf)/'
    r'(?P<slug>[^"/?#]+)(?:/(?P<tail>[^"#?]+))?)"',
    re.IGNORECASE,
)
ABS_DOC_LINK_RE = re.compile(
    r'https://docs\.redhat\.com/en/documentation/openshift_container_platform/'
    r'(?P<version>[^"/?#]+)/(?P<kind>html|html-single|pdf)/'
    r'(?P<slug>[^"/?#]+)(?:/(?P<tail>[^"#?]+))?',
    re.IGNORECASE,
)
PDF_HREF_RE = re.compile(
    r'(?P<href>(?:https://docs\.redhat\.com)?/en/documentation/'
    r'openshift_container_platform/(?P<version>[^"/?#]+)/pdf/'
    r'(?P<slug>[^"/?#]+)/(?P<file>[^"#?]+\.pdf))',
    re.IGNORECASE,
)

LOGGER = logging.getLogger("ocp-downloader")


@dataclass(frozen=True)
class CategorySpec:
    key: str
    group: str
    description: str
    slug_hints: tuple[str, ...]


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "overview": CategorySpec("overview", "core", "Platform overview and basics", ("overview",)),
    "architecture": CategorySpec("architecture", "core", "Architecture and components", ("architecture",)),
    "release_notes": CategorySpec("release_notes", "core", "Version changes and release notes", ("release_notes",)),
    "nodes": CategorySpec("nodes", "ops", "Node operations", ("nodes",)),
    "machine_management": CategorySpec(
        "machine_management",
        "ops",
        "Machine API and machine management",
        ("machine_management",),
    ),
    "cluster_observability_operator": CategorySpec(
        "cluster_observability_operator",
        "ops",
        "Cluster observability operator",
        ("cluster_observability_operator",),
    ),
    "monitoring": CategorySpec("monitoring", "ops", "Cluster monitoring", ("monitoring",)),
    "updating_clusters": CategorySpec(
        "updating_clusters",
        "ops",
        "Cluster update workflow",
        ("updating_clusters",),
    ),
    "disconnected_environments": CategorySpec(
        "disconnected_environments",
        "ops",
        "Disconnected and air-gapped operation",
        ("disconnected_environments",),
    ),
    "installing_on_bare_metal": CategorySpec(
        "installing_on_bare_metal",
        "install",
        "Bare metal installation",
        ("installing_on_bare_metal",),
    ),
    "installing_on_aws": CategorySpec(
        "installing_on_aws",
        "install",
        "AWS installation",
        ("installing_on_aws",),
    ),
    "installing_on_vmware_vsphere": CategorySpec(
        "installing_on_vmware_vsphere",
        "install",
        "VMware vSphere installation",
        ("installing_on_vmware_vsphere", "installing_on_vsphere"),
    ),
    "installing_on_azure": CategorySpec(
        "installing_on_azure",
        "install",
        "Azure installation",
        ("installing_on_azure",),
    ),
    "installing_on_google_cloud": CategorySpec(
        "installing_on_google_cloud",
        "install",
        "Google Cloud installation",
        ("installing_on_google_cloud",),
    ),
    "installing_on_openstack": CategorySpec(
        "installing_on_openstack",
        "install",
        "OpenStack installation",
        ("installing_on_openstack",),
    ),
    "installing_on_ibm_cloud": CategorySpec(
        "installing_on_ibm_cloud",
        "install",
        "IBM Cloud installation",
        ("installing_on_ibm_cloud",),
    ),
    "installing_on_any_platform": CategorySpec(
        "installing_on_any_platform",
        "install",
        "Generic install configuration",
        ("installing_on_any_platform",),
    ),
    "networking": CategorySpec(
        "networking",
        "network",
        "Core networking docs",
        ("networking_overview", "advanced_networking", "network_security"),
    ),
    "networking_overview": CategorySpec(
        "networking_overview",
        "network",
        "Networking overview",
        ("networking_overview",),
    ),
    "advanced_networking": CategorySpec(
        "advanced_networking",
        "network",
        "Advanced networking",
        ("advanced_networking",),
    ),
    "network_security": CategorySpec(
        "network_security",
        "network",
        "Network security",
        ("network_security",),
    ),
    "service_mesh": CategorySpec(
        "service_mesh",
        "network",
        "Service Mesh",
        ("service_mesh",),
    ),
    "storage": CategorySpec("storage", "storage", "Persistent storage", ("storage",)),
    "security_and_compliance": CategorySpec(
        "security_and_compliance",
        "security",
        "Security and compliance",
        ("security_and_compliance",),
    ),
    "authentication_and_authorization": CategorySpec(
        "authentication_and_authorization",
        "security",
        "Authentication, authorization, RBAC",
        ("authentication_and_authorization",),
    ),
    "building_applications": CategorySpec(
        "building_applications",
        "dev",
        "Application development and builds",
        ("building_applications",),
    ),
    "operators": CategorySpec("operators", "dev", "Operator usage and management", ("operators",)),
    "images": CategorySpec("images", "dev", "ImageStreams and image management", ("images",)),
    "cicd": CategorySpec("cicd", "cicd", "Tekton and CI/CD pipelines", ("cicd",)),
    "virtualization": CategorySpec("virtualization", "virtualization", "OpenShift Virtualization", ("virtualization",)),
    "lightspeed": CategorySpec(
        "lightspeed",
        "ai",
        "OpenShift Lightspeed",
        ("openshift_lightspeed", "lightspeed"),
    ),
    "openshift_lightspeed": CategorySpec(
        "openshift_lightspeed",
        "ai",
        "OpenShift Lightspeed",
        ("openshift_lightspeed",),
    ),
    "ai_workloads": CategorySpec(
        "ai_workloads",
        "ai",
        "AI workloads on OpenShift",
        ("ai_workloads",),
    ),
    "cloud": CategorySpec(
        "cloud",
        "cloud",
        "Managed or cloud install docs",
        (
            "installing_on_aws",
            "installing_on_azure",
            "installing_on_google_cloud",
            "installing_on_openstack",
            "installing_on_ibm_cloud",
        ),
    ),
}

PROFILES: dict[str, list[str]] = {
    "minimal": ["overview", "architecture", "storage", "networking", "security_and_compliance"],
    "core": [
        "overview",
        "architecture",
        "release_notes",
        "nodes",
        "machine_management",
        "networking",
        "storage",
        "security_and_compliance",
        "authentication_and_authorization",
    ],
    "ops": [
        "overview",
        "architecture",
        "nodes",
        "machine_management",
        "cluster_observability_operator",
        "monitoring",
        "updating_clusters",
        "disconnected_environments",
        "networking",
        "storage",
        "security_and_compliance",
    ],
    "rag": [
        "overview",
        "architecture",
        "release_notes",
        "nodes",
        "machine_management",
        "networking",
        "storage",
        "security_and_compliance",
        "authentication_and_authorization",
        "building_applications",
        "operators",
        "cicd",
        "lightspeed",
        "virtualization",
        "installing_on_bare_metal",
        "installing_on_vmware_vsphere",
        "disconnected_environments",
        "installing_on_aws",
    ],
    "install": [
        "overview",
        "installing_on_bare_metal",
        "installing_on_aws",
        "installing_on_vmware_vsphere",
        "installing_on_any_platform",
        "cloud",
    ],
    "full": list(CATEGORY_SPECS.keys()),
}

DEFAULT_VERSIONS = ["4.18", "4.21"]

TOKEN_TITLE_OVERRIDES = {
    "ai": "AI",
    "api": "API",
    "aws": "AWS",
    "cd": "CD",
    "ci": "CI",
    "cli": "CLI",
    "ibm": "IBM",
    "ocp": "OCP",
    "vm": "VM",
    "vsphere": "vSphere",
}

SLUG_TITLE_OVERRIDES = {
    "cicd": "CI_CD",
    "openshift_lightspeed": "OpenShift_Lightspeed",
    "ai_workloads": "AI_workloads",
}


@dataclass
class CatalogEntry:
    slug: str
    html_url: str | None = None
    html_single_url: str | None = None
    pdf_url: str | None = None


@dataclass
class DownloadTask:
    version: str
    requested_key: str
    slug: str
    url: str
    output_path: Path
    description: str
    group: str
    status: str = "pending"
    size_mb: float = 0.0
    error: str = ""


@dataclass
class DownloadReport:
    tasks: list[DownloadTask] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def done(self) -> list[DownloadTask]:
        return [task for task in self.tasks if task.status == "done"]

    @property
    def skipped(self) -> list[DownloadTask]:
        return [task for task in self.tasks if task.status == "skipped"]

    @property
    def failed(self) -> list[DownloadTask]:
        return [task for task in self.tasks if task.status == "failed"]

    @property
    def total_mb(self) -> float:
        return sum(task.size_mb for task in self.done)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def iter_doc_links(html: str) -> Iterable[tuple[str, str, str, str | None]]:
    for regex in (DOC_LINK_RE, ABS_DOC_LINK_RE):
        for match in regex.finditer(html):
            yield (
                match.group("version"),
                match.group("kind"),
                match.group("slug"),
                match.group("tail"),
            )


def parse_catalog_from_html(version: str, html: str) -> dict[str, CatalogEntry]:
    catalog: dict[str, CatalogEntry] = {}
    for link_version, kind, slug, tail in iter_doc_links(html):
        if link_version != version:
            continue
        entry = catalog.setdefault(slug, CatalogEntry(slug=slug))
        if kind == "html":
            entry.html_url = f"{BASE_URL}/{version}/html/{slug}/"
        elif kind == "html-single":
            entry.html_single_url = f"{BASE_URL}/{version}/html-single/{slug}/"
        elif kind == "pdf" and tail and tail.lower().endswith(".pdf"):
            entry.pdf_url = f"{BASE_URL}/{version}/pdf/{slug}/{tail}"
    return catalog


def merge_catalog_entries(base: dict[str, CatalogEntry], extra: dict[str, CatalogEntry]) -> dict[str, CatalogEntry]:
    merged = dict(base)
    for slug, entry in extra.items():
        current = merged.get(slug)
        if current is None:
            merged[slug] = entry
            continue
        merged[slug] = CatalogEntry(
            slug=slug,
            html_url=current.html_url or entry.html_url,
            html_single_url=current.html_single_url or entry.html_single_url,
            pdf_url=current.pdf_url or entry.pdf_url,
        )
    return merged


def build_title_candidates(slug: str) -> list[str]:
    candidates: list[str] = []

    if slug in SLUG_TITLE_OVERRIDES:
        candidates.append(SLUG_TITLE_OVERRIDES[slug])

    tokens = re.split(r"[_-]+", slug)
    normalized = "_".join(
        TOKEN_TITLE_OVERRIDES.get(token.casefold(), token.capitalize())
        for token in tokens
        if token
    )
    if normalized:
        candidates.append(normalized)

    simple = "_".join(token.capitalize() for token in tokens if token)
    if simple:
        candidates.append(simple)

    raw = slug.replace("-", "_")
    if raw:
        candidates.append(raw)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def build_pdf_guess(version: str, slug: str, title: str) -> str:
    filename = f"{PRODUCT_NAME}-{version}-{title}-en-US.pdf"
    return f"{BASE_URL}/{version}/pdf/{slug}/{filename}"


def resolve_profile_keys(
    profile: str,
    extra: list[str] | None,
    exclude: list[str] | None,
    custom: list[str] | None,
) -> list[str]:
    if custom:
        selected = list(custom)
    else:
        selected = list(PROFILES.get(profile, PROFILES["core"]))
        if extra:
            selected.extend(extra)

    excluded = set(exclude or [])
    deduped: list[str] = []
    seen: set[str] = set()
    for key in selected:
        if key in excluded:
            continue
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def resolve_slug_candidates(requested_key: str, catalog: dict[str, CatalogEntry]) -> list[str]:
    if requested_key in catalog:
        return [requested_key]

    spec = CATEGORY_SPECS.get(requested_key)
    if spec is None:
        normalized = requested_key.replace("-", "_").casefold()
        exact = [slug for slug in catalog if slug.casefold() == normalized]
        if exact:
            return exact
        fuzzy = [
            slug for slug in sorted(catalog)
            if normalized == slug.casefold() or normalized in slug.casefold()
        ]
        return fuzzy

    resolved: list[str] = []
    seen: set[str] = set()
    for hint in spec.slug_hints:
        if hint in catalog and hint not in seen:
            resolved.append(hint)
            seen.add(hint)
            continue
        for slug in sorted(catalog):
            lowered_slug = slug.casefold()
            lowered_hint = hint.casefold()
            if lowered_slug == lowered_hint or lowered_slug.startswith(lowered_hint + "_"):
                if slug not in seen:
                    resolved.append(slug)
                    seen.add(slug)
    return resolved


async def fetch_text(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        response = await client.get(url, timeout=60)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as exc:
        LOGGER.debug("failed to fetch %s: %s", url, exc)
        return None


async def discover_version_catalog(client: httpx.AsyncClient, version: str) -> dict[str, CatalogEntry]:
    urls = [
        f"{BASE_URL}/{version}",
        f"{BASE_URL}/{version}/html",
    ]
    catalog: dict[str, CatalogEntry] = {}
    for url in urls:
        html = await fetch_text(client, url)
        if not html:
            continue
        catalog = merge_catalog_entries(catalog, parse_catalog_from_html(version, html))
    return catalog


async def probe_pdf_url(client: httpx.AsyncClient, url: str) -> bool:
    try:
        response = await client.head(url, timeout=30, follow_redirects=True)
        if response.status_code < 400:
            content_type = response.headers.get("content-type", "").lower()
            return "pdf" in content_type or "octet-stream" in content_type
        if response.status_code not in {403, 405}:
            return False
    except httpx.HTTPError:
        pass

    try:
        response = await client.get(url, timeout=30, follow_redirects=True, headers={"Range": "bytes=0-0"})
        if response.status_code >= 400:
            return False
        content_type = response.headers.get("content-type", "").lower()
        return "pdf" in content_type or "octet-stream" in content_type
    except httpx.HTTPError:
        return False


async def resolve_pdf_url(
    client: httpx.AsyncClient,
    version: str,
    slug: str,
    entry: CatalogEntry,
) -> str | None:
    if entry.pdf_url:
        return entry.pdf_url

    for page_url in (entry.html_url, entry.html_single_url):
        if not page_url:
            continue
        html = await fetch_text(client, page_url)
        if not html:
            continue
        for match in PDF_HREF_RE.finditer(html):
            if match.group("version") != version:
                continue
            if match.group("slug") != slug:
                continue
            href = match.group("href")
            return href if href.startswith("https://") else urljoin("https://docs.redhat.com", href)

    for title in build_title_candidates(slug):
        guess = build_pdf_guess(version, slug, title)
        if await probe_pdf_url(client, guess):
            return guess

    return None


def build_output_path(output_base: Path, version: str, pdf_url: str) -> Path:
    parsed = urlparse(pdf_url)
    filename = Path(parsed.path).name
    return output_base / f"ocp-{version}" / filename


async def build_tasks_for_version(
    client: httpx.AsyncClient,
    version: str,
    requested_keys: list[str],
    output_base: Path,
) -> tuple[list[DownloadTask], dict[str, CatalogEntry]]:
    catalog = await discover_version_catalog(client, version)
    if not catalog:
        LOGGER.warning("version %s catalog discovery returned no categories", version)

    tasks: list[DownloadTask] = []
    seen_slugs: set[str] = set()

    for requested_key in requested_keys:
        spec = CATEGORY_SPECS.get(requested_key)
        description = spec.description if spec else requested_key
        group = spec.group if spec else "custom"

        resolved_slugs = resolve_slug_candidates(requested_key, catalog)
        if not resolved_slugs:
            LOGGER.warning("version %s: no matching slug found for %s", version, requested_key)
            continue

        for slug in resolved_slugs:
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            entry = catalog.get(slug, CatalogEntry(slug=slug))
            pdf_url = await resolve_pdf_url(client, version, slug, entry)
            if not pdf_url:
                LOGGER.warning("version %s: could not resolve PDF URL for slug %s", version, slug)
                continue
            tasks.append(
                DownloadTask(
                    version=version,
                    requested_key=requested_key,
                    slug=slug,
                    url=pdf_url,
                    output_path=build_output_path(output_base, version, pdf_url),
                    description=description,
                    group=group,
                )
            )

    return tasks, catalog


async def download_one(
    client: httpx.AsyncClient,
    task: DownloadTask,
    semaphore: asyncio.Semaphore,
) -> DownloadTask:
    async with semaphore:
        if task.output_path.exists() and task.output_path.stat().st_size > 1024:
            task.status = "skipped"
            task.size_mb = task.output_path.stat().st_size / (1024 * 1024)
            LOGGER.info("[SKIP] %s", task.output_path.name)
            return task

        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        task.status = "downloading"

        for attempt, delay in enumerate(RETRY_DELAYS + [0], start=1):
            try:
                LOGGER.info("[DOWN] v%s %s", task.version, task.slug)
                response = await client.get(task.url, timeout=180, follow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and "octet-stream" not in content_type:
                    raise RuntimeError(f"unexpected content-type: {content_type}")
                task.output_path.write_bytes(response.content)
                task.size_mb = len(response.content) / (1024 * 1024)
                task.status = "done"
                LOGGER.info("[DONE] %s (%.1f MB)", task.output_path.name, task.size_mb)
                return task
            except Exception as exc:  # noqa: BLE001
                task.error = str(exc)
                if delay:
                    LOGGER.warning(
                        "[RETRY] v%s %s attempt %d failed: %s",
                        task.version,
                        task.slug,
                        attempt,
                        exc,
                    )
                    await asyncio.sleep(delay)

        task.status = "failed"
        LOGGER.error("[FAIL] v%s %s: %s", task.version, task.slug, task.error)
        return task


async def download_all(tasks: list[DownloadTask], workers: int) -> DownloadReport:
    report = DownloadReport(tasks=tasks, start_time=time.time())
    semaphore = asyncio.Semaphore(workers)
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        await asyncio.gather(*(download_one(client, task, semaphore) for task in tasks))
    report.end_time = time.time()
    return report


def write_manifest(output_dir: Path, tasks: list[DownloadTask]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    entries = [
        {
            "file_name": task.output_path.name,
            "source_path": str(task.output_path),
            "version_tag": task.version,
            "product": "OCP",
            "requested_key": task.requested_key,
            "slug": task.slug,
            "group": task.group,
            "description": task.description,
            "size_mb": round(task.size_mb, 2),
            "url": task.url,
            "status": task.status,
        }
        for task in tasks
        if task.status in {"done", "skipped"}
    ]
    manifest_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def print_supported_categories() -> None:
    grouped: dict[str, list[CategorySpec]] = {}
    for spec in CATEGORY_SPECS.values():
        grouped.setdefault(spec.group, []).append(spec)

    print("\n=== Supported Friendly Categories ===\n")
    for group in sorted(grouped):
        print(f"[{group}]")
        for spec in sorted(grouped[group], key=lambda item: item.key):
            print(f"  {spec.key:32s} {spec.description}")
        print()

    print("=== Profiles ===\n")
    for profile, keys in PROFILES.items():
        print(f"  {profile:12s} {', '.join(keys)}")
    print()


def print_discovered_catalog(version: str, catalog: dict[str, CatalogEntry]) -> None:
    print(f"\n=== Discovered categories for OCP {version} ===")
    for slug in sorted(catalog):
        entry = catalog[slug]
        pdf_state = "pdf" if entry.pdf_url else "-"
        html_state = "html" if entry.html_url or entry.html_single_url else "-"
        print(f"  {slug:40s} {html_state:4s} {pdf_state:4s}")
    print()


def print_report(report: DownloadReport) -> None:
    elapsed = report.end_time - report.start_time
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"done:    {len(report.done):3d} ({report.total_mb:.1f} MB)")
    print(f"skipped: {len(report.skipped):3d}")
    print(f"failed:  {len(report.failed):3d}")
    print(f"elapsed: {elapsed:.1f}s")
    if report.failed:
        print("\nFailed items:")
        for task in report.failed:
            print(f"  - v{task.version} {task.slug}: {task.error}")
    print("=" * 60)


async def plan_tasks(
    versions: list[str],
    requested_keys: list[str],
    output_base: Path,
    list_remote_categories: bool,
) -> tuple[list[DownloadTask], dict[str, dict[str, CatalogEntry]]]:
    catalogs: dict[str, dict[str, CatalogEntry]] = {}
    all_tasks: list[DownloadTask] = []

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        for version in versions:
            tasks, catalog = await build_tasks_for_version(client, version, requested_keys, output_base)
            catalogs[version] = catalog
            all_tasks.extend(tasks)
            if list_remote_categories:
                print_discovered_catalog(version, catalog)

    return all_tasks, catalogs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download curated OCP PDFs from docs.redhat.com",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS, help="OCP versions to target")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="rag", help="friendly category profile")
    parser.add_argument("--categories", nargs="+", default=None, help="explicit friendly keys or raw slugs")
    parser.add_argument("--extra", nargs="+", default=None, help="extra friendly keys to append")
    parser.add_argument("--exclude", nargs="+", default=None, help="friendly keys to exclude")
    parser.add_argument("--workers", type=int, default=4, help="concurrent downloads")
    parser.add_argument("--output", type=Path, default=None, help="base output directory")
    parser.add_argument("--dry-run", action="store_true", help="show resolved PDFs without downloading")
    parser.add_argument("--list-categories", action="store_true", help="show supported friendly categories")
    parser.add_argument(
        "--list-remote-categories",
        action="store_true",
        help="discover and print actual remote slugs for each requested version",
    )
    return parser


def main() -> None:
    ensure_utf8_console()
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.list_categories:
        print_supported_categories()
        return

    project_root = Path(__file__).resolve().parent.parent
    output_base = args.output or (project_root / "data" / "corpus" / "pdfs")
    requested_keys = resolve_profile_keys(args.profile, args.extra, args.exclude, args.categories)
    if not requested_keys:
        parser.error("no categories selected")

    tasks, catalogs = asyncio.run(
        plan_tasks(
            versions=args.versions,
            requested_keys=requested_keys,
            output_base=output_base,
            list_remote_categories=args.list_remote_categories,
        )
    )

    print("\n" + "=" * 60)
    print("OCP PDF Download Plan")
    print("=" * 60)
    print(f"versions:   {', '.join(args.versions)}")
    print(f"profile:    {args.profile}")
    print(f"requested:  {', '.join(requested_keys)}")
    print(f"resolved:   {len(tasks)} PDFs")
    print(f"output:     {output_base}")
    print("=" * 60 + "\n")

    if args.dry_run or args.list_remote_categories:
        for task in tasks:
            print(f"[v{task.version}] {task.slug:32s} -> {task.output_path.name}")
            print(f"  {task.url}")
        if args.dry_run:
            print(f"\nDry-run complete: {len(tasks)} PDF(s) resolved.")
        return

    report = asyncio.run(download_all(tasks, workers=args.workers))
    for version in args.versions:
        version_tasks = [task for task in report.tasks if task.version == version]
        if version_tasks:
            write_manifest(output_base / f"ocp-{version}", version_tasks)
    write_manifest(output_base, report.tasks)
    print_report(report)


if __name__ == "__main__":
    main()
