"""Import selected openshift-docs AsciiDoc sources as cleaned Markdown files.

Usage:
    python scripts/import_openshift_docs.py --repo-root C:\\src\\openshift-docs
    python scripts/import_openshift_docs.py --repo-root ../openshift-docs --section installing
"""
from __future__ import annotations

import argparse
import html
import os
import re
import stat
import shutil
from pathlib import Path


DEFAULT_SECTIONS = [
    "installing",
    "post_installation_configuration",
    "networking",
    "storage",
    "authentication",
]

DIRECTIVE_PREFIXES = (
    "ifdef::",
    "ifndef::",
    "endif::",
    "include::",
    "toc::",
    ":context:",
    ":_mod-docs-content-type:",
)

ATTRIBUTE_LINE_RE = re.compile(r"^:[^:\s][^:]*:\s*.*$")
HEADING_RE = re.compile(r"^(=+)\s+(.+)$")
ORDERED_LIST_RE = re.compile(r"^\.\s+(.+)$")
UNORDERED_LIST_RE = re.compile(r"^\*\s+(.+)$")
SOURCE_BLOCK_RE = re.compile(r"^\[source(?:%[^\],]+)?(?:,([a-zA-Z0-9_+-]+))?.*\]$")
TABLE_FENCE_RE = re.compile(r"^\|===$")
CALLOUT_RE = re.compile(r"\s*<\d+>\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, help="Local checkout path of openshift/openshift-docs")
    parser.add_argument("--version", default="4.20", help="Target OCP version tag used in output path. Default: 4.20")
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        help="Section directory under openshift-docs. Can be repeated. Default set focuses on install/deploy topics.",
    )
    parser.add_argument(
        "--output-root",
        default="data/corpus/pdfs",
        help="Base output root. Markdown files will be written under ocp-<version>-openshift-docs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the target output directory before importing.",
    )
    return parser.parse_args()


def selected_sections(args: argparse.Namespace) -> list[str]:
    sections = [str(section).strip() for section in args.section if str(section).strip()]
    return sections or DEFAULT_SECTIONS


def import_output_dir(output_root: Path, version: str) -> Path:
    return output_root / f"ocp-{version}-openshift-docs"


def collect_adoc_files(repo_root: Path, sections: list[str]) -> list[Path]:
    files: list[Path] = []
    repo_root = repo_root.resolve()
    for section in sections:
        section_dir = repo_root / section
        if not section_dir.exists():
            raise FileNotFoundError(f"openshift-docs section not found: {section_dir}")
        section_files = sorted(path for path in section_dir.rglob("*.adoc") if path.is_file())
        if not section_files:
            raise FileNotFoundError(f"No .adoc files found under: {section_dir}")
        files.extend(path.resolve() for path in section_files)
    return files


def _strip_front_attributes(lines: list[str]) -> list[str]:
    while lines:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        if ATTRIBUTE_LINE_RE.match(stripped):
            lines.pop(0)
            continue
        break
    return lines


def _clean_inline_adoc(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"link:[^\[]+\[([^\]]+)\]", r"\1", value)
    value = re.sub(r"xref:[^\[]+\[([^\]]+)\]", r"\1", value)
    value = re.sub(r"<<[^,>]+,([^>]+)>>", r"\1", value)
    value = re.sub(r"<<([^>]+)>>", r"\1", value)
    value = re.sub(r"image::[^\[]+\[([^\]]*)\]", r"\1", value)
    value = re.sub(r"pass:\[[^\]]*\]", "", value)
    value = re.sub(r"^\[(IMPORTANT|NOTE|TIP|WARNING|CAUTION)\]\s*$", "", value, flags=re.IGNORECASE)
    value = CALLOUT_RE.sub("", value)
    value = value.replace("`+", "`").replace("+`", "`")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _render_simple_table(table_lines: list[str]) -> list[str]:
    rows: list[list[str]] = []
    for raw_line in table_lines:
        stripped = raw_line.strip()
        if not stripped or TABLE_FENCE_RE.match(stripped):
            continue
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.split("|")[1:] if cell.strip()]
        if cells:
            rows.append([_clean_inline_adoc(cell) for cell in cells])

    if len(rows) < 2:
        return [" ".join(" ".join(row) for row in rows).strip()] if rows else []

    header = rows[0]
    body = rows[1:]
    markdown = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        padded = row + [""] * max(0, len(header) - len(row))
        markdown.append("| " + " | ".join(padded[: len(header)]) + " |")
    return markdown


def convert_adoc_to_markdown(adoc_text: str) -> str:
    lines = _strip_front_attributes(adoc_text.replace("\r\n", "\n").replace("\r", "\n").splitlines())
    output: list[str] = []
    in_code_block = False
    code_fence = ""
    code_lines: list[str] = []
    pending_code_lang = ""
    in_table = False
    table_lines: list[str] = []
    previous_blank = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if in_table:
            table_lines.append(line)
            if TABLE_FENCE_RE.match(stripped):
                rendered_table = _render_simple_table(table_lines)
                if rendered_table:
                    if output and output[-1] != "":
                        output.append("")
                    output.extend(rendered_table)
                    output.append("")
                table_lines = []
                in_table = False
                previous_blank = True
            continue

        source_match = SOURCE_BLOCK_RE.match(stripped)
        if source_match:
            pending_code_lang = (source_match.group(1) or "").strip().lower()
            continue

        if TABLE_FENCE_RE.match(stripped):
            in_table = True
            table_lines = [line]
            continue

        if stripped in {"----", "...."}:
            if in_code_block:
                output.append("```")
                output.extend(code_lines)
                output.append("```")
                output.append("")
                in_code_block = False
                code_lines = []
                code_fence = ""
                previous_blank = True
            else:
                code_fence = pending_code_lang
                output.append(f"```{code_fence}".rstrip())
                in_code_block = True
                previous_blank = False
            pending_code_lang = ""
            continue

        if in_code_block:
            code_lines.append(line.rstrip())
            continue

        if not stripped:
            if output and not previous_blank:
                output.append("")
                previous_blank = True
            continue

        if stripped.startswith("//") or ATTRIBUTE_LINE_RE.match(stripped):
            continue
        if any(stripped.startswith(prefix) for prefix in DIRECTIVE_PREFIXES):
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            level = min(len(heading_match.group(1)), 6)
            heading_text = _clean_inline_adoc(heading_match.group(2))
            if heading_text:
                if output and output[-1] != "":
                    output.append("")
                output.append("#" * level + " " + heading_text)
                output.append("")
                previous_blank = True
            continue

        ordered_match = ORDERED_LIST_RE.match(stripped)
        if ordered_match:
            output.append("1. " + _clean_inline_adoc(ordered_match.group(1)))
            previous_blank = False
            continue

        unordered_match = UNORDERED_LIST_RE.match(stripped)
        if unordered_match:
            output.append("- " + _clean_inline_adoc(unordered_match.group(1)))
            previous_blank = False
            continue

        if stripped.startswith("NOTE:") or stripped.startswith("TIP:") or stripped.startswith("IMPORTANT:") or stripped.startswith("WARNING:") or stripped.startswith("CAUTION:"):
            output.append("> " + _clean_inline_adoc(stripped))
            previous_blank = False
            continue

        cleaned = _clean_inline_adoc(stripped)
        if cleaned:
            output.append(cleaned)
            previous_blank = False

    if in_code_block:
        output.append("```")
        output.extend(code_lines)
        output.append("```")

    while output and not output[-1].strip():
        output.pop()
    return "\n".join(output).strip() + "\n"


def build_output_path(repo_root: Path, source_path: Path, output_dir: Path) -> Path:
    relative = source_path.resolve().relative_to(repo_root.resolve())
    return (output_dir / relative).with_suffix(".md")


def _remove_tree(path: Path) -> None:
    def _onexc(func, target, exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
        except OSError:
            pass
        func(target)

    shutil.rmtree(path, onexc=_onexc)


def import_openshift_docs(repo_root: Path, output_dir: Path, sections: list[str], *, clean: bool = False) -> list[Path]:
    if clean and output_dir.exists():
        _remove_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for source_path in collect_adoc_files(repo_root, sections):
        markdown_text = convert_adoc_to_markdown(source_path.read_text(encoding="utf-8", errors="ignore"))
        if not markdown_text.strip():
            continue
        output_path = build_output_path(repo_root, source_path, output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_text, encoding="utf-8")
        created.append(output_path)
    return created


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = import_output_dir(Path(args.output_root), args.version)
    sections = selected_sections(args)
    created = import_openshift_docs(repo_root, output_dir, sections, clean=args.clean)
    print("\n".join(str(path) for path in created))


if __name__ == "__main__":
    main()
