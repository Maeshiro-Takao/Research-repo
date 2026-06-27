#!/usr/bin/env python3
"""MBRACE 競走成績 (.lzh) を取得し TXT を 元データ/競走成績/{N}年/{M}月/ に保存する。"""

from __future__ import annotations

import argparse
import calendar
import io
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lhafile

from normalize_race_txt import normalize_text

BASE_URL = "https://www1.mbrace.or.jp/od2/K"
DEFAULT_OUT = Path(__file__).resolve().parent / "競走成績"


def iter_dates(year: int):
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            yield year, month, day


def lzh_url(year: int, month: int, day: int) -> str:
    yymm = f"{year % 100:02d}{month:02d}{day:02d}"
    return f"{BASE_URL}/{year}{month:02d}/k{yymm}.lzh"


def txt_path(out_root: Path, year: int, month: int, day: int) -> Path:
    yy = year % 100
    name = f"K{yy:02d}{month:02d}{day:02d}.TXT"
    return out_root / f"{yy}年" / f"{month}月" / name


def download_and_extract(url: str, dest: Path, timeout: float = 30.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()

    arc = lhafile.LhaFile(io.BytesIO(data))
    names = arc.namelist()
    if not names:
        raise ValueError(f"LZH にファイルがありません: {url}")

    content = arc.read(names[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = content.decode("cp932")
    dest.write_text(normalize_text(text), encoding="utf-8")


def fetch_one(year: int, month: int, day: int, out_root: Path) -> str:
    dest = txt_path(out_root, year, month, day)
    url = lzh_url(year, month, day)
    try:
        download_and_extract(url, dest)
        return "saved"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing"
        return f"error:http{e.code}"
    except Exception as e:
        return f"error:{e}"


def download_years(
    years: list[int],
    out_root: Path,
    *,
    workers: int = 8,
    skip_existing: bool = True,
) -> dict[str, int]:
    stats = {"saved": 0, "skipped": 0, "missing": 0, "error": 0}
    tasks: list[tuple[int, int, int]] = []

    for year in years:
        for y, month, day in iter_dates(year):
            dest = txt_path(out_root, y, month, day)
            if skip_existing and dest.exists():
                stats["skipped"] += 1
                continue
            tasks.append((y, month, day))

    print(f"DL対象: {len(tasks)} 日 (skip={stats['skipped']})", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(fetch_one, y, m, d, out_root): (y, m, d)
            for y, m, d in tasks
        }
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            if result == "saved":
                stats["saved"] += 1
            elif result == "missing":
                stats["missing"] += 1
            elif result == "skipped":
                stats["skipped"] += 1
            else:
                stats["error"] += 1
                y, m, d = futures[fut]
                print(f"  ERROR K{y % 100:02d}{m:02d}{d:02d}: {result}", flush=True)

            done += 1
            if done % 50 == 0:
                print(
                    f"  progress {done}/{len(tasks)} "
                    f"(saved={stats['saved']} missing={stats['missing']})",
                    flush=True,
                )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="MBRACE 競走成績 LZH を DL して TXT 保存")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2012, 2013],
        help="取得する西暦年 (default: 2012 2013)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="保存先ルート (default: 元データ/競走成績)",
    )
    parser.add_argument("--workers", type=int, default=8, help="並列数")
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存 TXT があっても再取得",
    )
    args = parser.parse_args()

    print(f"保存先: {args.out}")
    print(f"対象年: {args.years}")
    stats = download_years(
        args.years,
        args.out,
        workers=args.workers,
        skip_existing=not args.force,
    )
    print(
        f"\n完了: saved={stats['saved']} skipped={stats['skipped']} "
        f"missing={stats['missing']} error={stats['error']}"
    )
    return 1 if stats["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
