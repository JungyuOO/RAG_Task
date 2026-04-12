"""고객사 메뉴얼(.md) 파일을 재인덱싱하는 스크립트.

upload API를 사용하므로 기존 청크가 upsert로 교체됩니다 (파일 삭제 없음).

사용법:
    python scripts/reindex_manuals.py [--base-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path


def get_manual_files(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/api/library") as resp:
        docs = json.load(resp).get("indexed_documents", [])
    return [d["file_name"] for d in docs if d.get("doc_type") == "operation_manual"]


def reindex_file(base_url: str, file_path: Path) -> None:
    boundary = "----ManualReindexBoundary"
    file_bytes = file_path.read_bytes()
    # 서버에서 generated/ 하위로 저장되도록 filename에 경로 포함하지 않음
    # (routes_shared.py가 .md 파일을 자동으로 generated/에 저장)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{file_path.name}"\r\n'
        f"Content-Type: text/markdown\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/library/upload",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            t = ev.get("type")
            if t == "file_indexed":
                print(f"  ✓ {ev.get('file')}  chunks={ev.get('indexed_chunks')}  pages={ev.get('indexed_pages')}")
            elif t == "file_error":
                print(f"  ✗ {ev.get('file')}: {ev.get('error')}")
            elif t == "done":
                print(f"  → total_chunks={ev.get('total_chunks')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"=== 고객사 메뉴얼 재인덱싱 ({base_url}) ===\n")

    manual_names = get_manual_files(base_url)
    print(f"대상 파일 {len(manual_names)}개\n")
    if not manual_names:
        print("재인덱싱할 파일 없음.")
        return

    generated_dir = Path("data/corpus/pdfs/generated")
    for name in manual_names:
        file_path = generated_dir / name
        if not file_path.exists():
            print(f"  스킵: {name} (로컬 파일 없음)")
            continue
        print(f"  {name} ...")
        try:
            reindex_file(base_url, file_path)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            print(f"  ✗ HTTP {exc.code}: {body[:200]}")
        except Exception as exc:
            print(f"  ✗ {exc}")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
