"""
丸亀競走成績TXTの [払戻金] セクションから3連単とオッズを抽出する。

各レース1行: 開催日, 日目, レース, 3連単, 3連単オッズ

[払戻金] 形式例:
  [払戻金] ３連単 ３連複 ２連単 ２連複
  1R 4-6-3 71040 3-4-6 10690 4-6 17320 4-6 6570

使い方:
    python オッズ抽出コード/ExtOdds.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "元データ" / "競走成績"
OUTPUT_DIR = BASE_DIR / "オッズデータ"
MARUGAME_CODE = "15"

COLUMNS = ["開催日", "日目", "レース", "3連単", "3連単オッズ"]

DATASETS = [
    {"years": range(14, 25), "output": "丸亀学習用_3連単オッズ.csv"},
    {"years": range(25, 26), "output": "丸亀テスト用_3連単オッズ.csv"},
]

DAY_PATTERN = re.compile(r"第\s*(\d+)日")
PAYOUT_HEADER = "[払戻金]"
PAYOUT_RACE = re.compile(r"^(\d+)R\s+(\S+)\s+(\S+)")
RACE_DETAIL = re.compile(r"^\d+R\s+")


def extract_date_from_filename(filename: str) -> str | None:
    m = re.match(r"K(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        return None
    yy, mm, dd = m.groups()
    return f"{2000 + int(yy)}-{int(mm):02d}-{int(dd):02d}"


def file_sort_key(path: Path) -> tuple[int, int, int]:
    m = re.match(r"K(\d{2})(\d{2})(\d{2})", path.name)
    if not m:
        return (9999, 99, 99)
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def extract_day_number(block_lines: list[str]) -> int | None:
    for line in block_lines:
        m = DAY_PATTERN.search(line)
        if m:
            return int(m.group(1))
    return None


def parse_payout_line(line: str) -> tuple[int, str, float] | None:
    """1R 4-6-3 71040 ... 形式（[払戻金]表）をパース"""
    m = PAYOUT_RACE.match(line)
    if not m:
        return None

    race_no = int(m.group(1))
    combo_str = m.group(2)
    payout_str = m.group(3)

    if combo_str == "不成立" or not re.fullmatch(r"\d-\d-\d", combo_str):
        return None

    try:
        payout = int(payout_str)
    except ValueError:
        return None

    # 100円あたりの払戻 → 倍率（オッズ）
    return race_no, combo_str, payout / 100.0


def process_marugame_block(block_lines: list[str], current_date: str) -> list[dict]:
    day_number = extract_day_number(block_lines)
    if day_number is None or current_date is None:
        return []

    records: list[dict] = []
    in_payout = False

    for line in block_lines:
        if not line:
            continue

        if line.startswith(PAYOUT_HEADER):
            in_payout = True
            continue

        if not in_payout:
            continue

        parsed = parse_payout_line(line)
        if parsed is None:
            if RACE_DETAIL.match(line):
                in_payout = False
            continue

        race_no, combo_str, odds = parsed
        records.append({
            "開催日": current_date,
            "日目": day_number,
            "レース": race_no,
            "3連単": combo_str,
            "3連単オッズ": odds,
        })

    return records


def extract_records(years: range) -> list[dict]:
    records: list[dict] = []

    for year in years:
        year_dir = INPUT_DIR / f"{year}年"
        print("processing:", year_dir)
        if not year_dir.exists():
            continue

        for txt_file in sorted(year_dir.rglob("*.TXT"), key=file_sort_key):
            current_date = extract_date_from_filename(txt_file.name)
            with open(txt_file, encoding="UTF-8") as f:
                lines = f.readlines()

            in_block = False
            block_lines: list[str] = []

            for raw in lines:
                s = raw.strip()
                if s == f"{MARUGAME_CODE}KBGN":
                    in_block = True
                    block_lines = []
                    continue
                if s == f"{MARUGAME_CODE}KEND":
                    if block_lines:
                        records.extend(process_marugame_block(block_lines, current_date))
                    in_block = False
                    block_lines = []
                    continue
                if in_block:
                    block_lines.append(s)

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n=== {dataset['output']} ===")
        records = extract_records(dataset["years"])
        df = pd.DataFrame(records, columns=COLUMNS)

        output_path = OUTPUT_DIR / dataset["output"]
        df.to_csv(output_path, index=False, encoding="UTF-8-sig")

        print("総件数:", len(df))
        if len(df) > 0:
            print("開催日数:", df["開催日"].nunique())
            print(
                "3連単オッズ: "
                f"min={df['3連単オッズ'].min():.1f}, "
                f"median={df['3連単オッズ'].median():.1f}, "
                f"max={df['3連単オッズ'].max():.1f}"
            )
        print("保存:", output_path)


if __name__ == "__main__":
    main()
