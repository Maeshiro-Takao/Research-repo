#!/usr/bin/env python3
"""競走成績 TXT の余分な空白を 14年形式に揃える。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent / "競走成績"


def normalize_line(line: str) -> str:
    text = line.rstrip("\r\n")
    if not text.strip():
        return ""
    text = text.strip().replace("\u3000", " ")
    return re.sub(r" +", " ", text)


def normalize_text(content: str) -> str:
    lines = content.splitlines()
    return "\n".join(normalize_line(line) for line in lines) + "\n"


def normalize_file(path: Path, *, dry_run: bool = False) -> bool:
    raw = path.read_bytes()
    for enc in ("utf-8", "cp932"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("unknown", b"", 0, 1, path.name)

    current = text if text.endswith("\n") else text + "\n"
    normalized = normalize_text(text)
    if normalized == current:
        return False

    if not dry_run:
        path.write_text(normalized, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="競走成績 TXT の空白を正規化")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[12, 13],
        help="対象年 (default: 12 13)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="競走成績ルート",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = 0
    total = 0
    for year in args.years:
        year_dir = args.dir / f"{year}年"
        if not year_dir.exists():
            print(f"skip: {year_dir} not found", file=sys.stderr)
            continue
        for path in sorted(year_dir.rglob("*.TXT")):
            total += 1
            if normalize_file(path, dry_run=args.dry_run):
                changed += 1

    action = "would change" if args.dry_run else "changed"
    print(f"{action}: {changed}/{total} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
