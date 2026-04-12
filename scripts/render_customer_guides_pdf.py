from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import fitz


_TABLE_SEPARATOR_RE = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render generated customer guide markdown files into PDF.")
    parser.add_argument(
        "--input-root",
        default="data/corpus/pdfs/generated",
        help="Directory containing generated markdown files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/corpus/pdfs/generated_pdf",
        help="Directory where rendered PDF files will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.md",
        help="Glob pattern for markdown files.",
    )
    return parser.parse_args()


def _strip_front_matter(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return markdown_text

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1 :])
    return markdown_text


def _parse_markdown_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _render_markdown_table(lines: list[str]) -> str:
    rows = [_parse_markdown_table_row(line) for line in lines]
    if len(rows) < 2:
        return f"<p>{html.escape(' '.join(lines))}</p>"

    headers = rows[0]
    body_rows = rows[2:]
    thead = "<tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in headers) + "</tr>"
    tbody_rows = []
    for row in body_rows:
        padded = row + [""] * max(0, len(headers) - len(row))
        tbody_rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in padded[: len(headers)]) + "</tr>")
    tbody = "".join(tbody_rows)
    return f"<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>"


def markdown_to_blocks(markdown_text: str, title: str) -> list[str]:
    lines = _strip_front_matter(markdown_text).splitlines()
    blocks: list[str] = [f"<div class='meta'>Generated customer handbook PDF export</div><h1>{html.escape(title)}</h1>"]

    in_code = False
    code_lines: list[str] = []
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("---") and not in_code:
            blocks.append("<hr>")
            index += 1
            continue

        if stripped.startswith("```"):
            if in_code:
                blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if not stripped:
            index += 1
            continue
        if stripped.startswith("# "):
            blocks.append(f"<h1>{html.escape(stripped[2:].strip())}</h1>")
            index += 1
            continue
        if stripped.startswith("## "):
            blocks.append(f"<h2>{html.escape(stripped[3:].strip())}</h2>")
            index += 1
            continue
        if stripped.startswith("### "):
            blocks.append(f"<h3>{html.escape(stripped[4:].strip())}</h3>")
            index += 1
            continue
        if stripped.startswith(">"):
            blocks.append(f"<blockquote>{html.escape(stripped.lstrip('>').strip())}</blockquote>")
            index += 1
            continue
        if (
            "|" in stripped
            and index + 1 < len(lines)
            and _TABLE_SEPARATOR_RE.match(lines[index + 1].strip())
        ):
            table_lines = [line]
            index += 1
            while index < len(lines):
                candidate = lines[index].rstrip("\n")
                candidate_stripped = candidate.strip()
                if not candidate_stripped:
                    break
                if "|" not in candidate_stripped:
                    break
                table_lines.append(candidate)
                index += 1
            blocks.append(_render_markdown_table(table_lines))
            continue
        if re.match(r"^\s*[-*]\s+", line):
            items: list[str] = []
            while index < len(lines) and re.match(r"^\s*[-*]\s+", lines[index]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[index]).strip())
                index += 1
            blocks.append("<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items: list[str] = []
            while index < len(lines) and re.match(r"^\s*\d+\.\s+", lines[index]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[index]).strip())
                index += 1
            blocks.append("<ol>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ol>")
            continue
        blocks.append(f"<p>{html.escape(line)}</p>")
        index += 1

    if code_lines:
        blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")

    return blocks


def render_blocks_to_pdf(output_path: Path, blocks: list[str]) -> None:
    css = """
    body { font-family: 'Malgun Gothic','Apple SD Gothic Neo','Segoe UI',sans-serif; color: #17212b; line-height: 1.7; }
    h1 { font-size: 26px; border-bottom: 2px solid #dbe3f2; padding-bottom: 8px; margin-top: 0; }
    h2 { font-size: 20px; margin-top: 28px; color: #123a7c; }
    h3 { font-size: 16px; margin-top: 22px; color: #23406f; }
    p, li { font-size: 12.5px; }
    pre { background: #0f172a; color: #e2e8f0; padding: 14px 16px; border-radius: 10px; font-size: 11px; line-height: 1.55; white-space: pre-wrap; }
    code { font-family: 'Consolas','SFMono-Regular',monospace; }
    blockquote { border-left: 4px solid #c9d7f0; padding: 8px 14px; background: #f8fbff; color: #42526a; }
    ul, ol { padding-left: 22px; }
    table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 11.5px; }
    th, td { border: 1px solid #d9e2f0; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #eef4ff; color: #16335f; font-weight: 700; }
    hr { border: none; border-top: 1px solid #d9e2f0; margin: 24px 0; }
    .meta { font-size: 11px; color: #6b7a90; margin-bottom: 18px; }
    """
    html_text = (
        "<html><head><meta charset='utf-8'><style>"
        + css
        + "</style></head><body>"
        + "".join(blocks)
        + "</body></html>"
    )
    story = fitz.Story(html_text, user_css=css)
    writer = fitz.DocumentWriter(str(output_path))
    mediabox = fitz.Rect(0, 0, 595, 842)
    content_rect = fitz.Rect(36, 36, 559, 806)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _filled = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()


def convert_markdown_file(path: Path, output_root: Path) -> Path:
    markdown_text = path.read_text(encoding="utf-8")
    title = path.stem
    blocks = markdown_to_blocks(markdown_text, title)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{path.stem}.pdf"
    render_blocks_to_pdf(output_path, blocks)
    return output_path


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    files = sorted(input_root.glob(args.pattern))
    if not files:
        raise SystemExit("No markdown files found to convert.")

    created: list[str] = []
    for path in files:
        created.append(str(convert_markdown_file(path, output_root)))
    print("\n".join(created))


if __name__ == "__main__":
    main()
