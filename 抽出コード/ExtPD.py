"""
元データ/選手データ の固定長テキスト（ファン手帳）を CSV に変換する。

レイアウト: https://www.boatrace.jp/owpc/pc/extra/data/layout.html
1レコード = 416バイト（CP932）

使い方:
    python 抽出コード/ExtPD.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "元データ" / "選手データ"
OUTPUT_DIR = BASE_DIR / "編集データ"

DATASETS = [
    {"years": range(14, 25), "output": "丸亀学習用_選手データ.csv"},
    {"years": range(25, 26), "output": "丸亀テスト用_選手データ.csv"},
]

OUTPUT_COLUMNS = [
    "登番", "名前漢字", "級", "身長", "体重", "勝率", "複勝率",
    "1着回数", "2着回数", "出走回数", "優出回数", "優勝回数", "平均スタートタイミング",
]
for _course in range(1, 7):
    OUTPUT_COLUMNS += [
        f"{_course}コース進入回数",
        f"{_course}コース複勝率",
        f"{_course}コース平均スタートタイミング",
        f"{_course}コース平均スタート順位",
    ]
OUTPUT_COLUMNS += [
    "前期級", "前々期級", "前々々期級", "前期能力指数", "今期能力指数",
    "年", "期", "算出期間自", "算出期間至", "養成期",
]
for _course in range(1, 7):
    for _rank in range(1, 7):
        OUTPUT_COLUMNS.append(f"{_course}コース{_rank}着回数")
OUTPUT_COLUMNS += ["出身地", "生年月日_変換"]

# F/L/K/S・欠場系はレースデータ側でも除外しているため出力しない
EXCLUDED_RESULT_CODES = ("F", "L0", "L1", "K0", "K1", "S0", "S1", "S2")
SKIP_FIELDS = {
    "名前カナ", "支部", "年号", "生年月日", "性別", "年齢", "血液型",
    "コースなしL0回数", "コースなしL1回数", "コースなしK0回数", "コースなしK1回数",
}
for _course in range(1, 7):
    for _code in EXCLUDED_RESULT_CODES:
        SKIP_FIELDS.add(f"{_course}コース{_code}回数")

# (項目名, バイト数)
FIELD_SPEC: list[tuple[str, int]] = [
    ("登番", 4),
    ("名前漢字", 16),
    ("名前カナ", 15),
    ("支部", 4),
    ("級", 2),
    ("年号", 1),
    ("生年月日", 6),
    ("性別", 1),
    ("年齢", 2),
    ("身長", 3),
    ("体重", 2),
    ("血液型", 2),
    ("勝率", 4),
    ("複勝率", 4),
    ("1着回数", 3),
    ("2着回数", 3),
    ("出走回数", 3),
    ("優出回数", 2),
    ("優勝回数", 2),
    ("平均スタートタイミング", 3),
]
for course in range(1, 7):
    FIELD_SPEC += [
        (f"{course}コース進入回数", 3),
        (f"{course}コース複勝率", 4),
        (f"{course}コース平均スタートタイミング", 3),
        (f"{course}コース平均スタート順位", 3),
    ]
FIELD_SPEC += [
    ("前期級", 2),
    ("前々期級", 2),
    ("前々々期級", 2),
    ("前期能力指数", 4),
    ("今期能力指数", 4),
    ("年", 4),
    ("期", 1),
    ("算出期間自", 8),
    ("算出期間至", 8),
    ("養成期", 3),
]
for course in range(1, 7):
    for rank in range(1, 7):
        FIELD_SPEC.append((f"{course}コース{rank}着回数", 3))
    for code in ("F", "L0", "L1", "K0", "K1", "S0", "S1", "S2"):
        FIELD_SPEC.append((f"{course}コース{code}回数", 2))
FIELD_SPEC += [
    ("コースなしL0回数", 2),
    ("コースなしL1回数", 2),
    ("コースなしK0回数", 2),
    ("コースなしK1回数", 2),
    ("出身地", 6),
]

RECORD_SIZE = sum(size for _, size in FIELD_SPEC)
assert RECORD_SIZE == 416

RATE_2DEC = {"勝率", "平均スタートタイミング", "前期能力指数", "今期能力指数"}
RATE_2DEC |= {f"{c}コース平均スタートタイミング" for c in range(1, 7)}
RATE_2DEC |= {f"{c}コース平均スタート順位" for c in range(1, 7)}

RATE_1DEC = {"複勝率"}
RATE_1DEC |= {f"{c}コース複勝率" for c in range(1, 7)}

INT_FIELDS = {
    "登番", "年齢", "身長", "体重", "1着回数", "2着回数", "出走回数",
    "優出回数", "優勝回数", "年", "養成期",
}
INT_FIELDS |= {f"{c}コース進入回数" for c in range(1, 7)}
for course in range(1, 7):
    for rank in range(1, 7):
        INT_FIELDS.add(f"{course}コース{rank}着回数")


def parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or not re.fullmatch(r"\d+", value):
        return None
    return int(value)


def parse_rate(value: str, decimals: int) -> float | None:
    num = parse_int(value)
    if num is None:
        return None
    return num / (10 ** decimals)


def era_to_year(era: str, yy: int) -> int | None:
    if era == "S":
        return 1925 + yy
    if era == "H":
        return 1988 + yy
    if era == "R":
        return 2018 + yy
    return None


def parse_birth_date(era: str, yymmdd: str) -> str | None:
    yymmdd = yymmdd.strip()
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = era_to_year(era.strip(), yy)
    if year is None or mm < 1 or mm > 12 or dd < 1 or dd > 31:
        return None
    return f"{year}-{mm:02d}-{dd:02d}"


def parse_yyyymmdd(value: str) -> str | None:
    value = value.strip()
    if len(value) != 8 or not value.isdigit():
        return None
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def convert_field(name: str, raw: str) -> str | int | float | None:
    value = raw.strip()
    if name in {"名前漢字", "名前カナ", "支部", "級", "年号", "血液型", "出身地"}:
        return value or None
    if name in {"前期級", "前々期級", "前々々期級"}:
        return value or None
    if name == "性別":
        return value if value in {"1", "2"} else value or None
    if name == "期":
        return parse_int(value)
    if name == "生年月日":
        return value
    if name in {"算出期間自", "算出期間至"}:
        return parse_yyyymmdd(value)
    if name in RATE_2DEC:
        return parse_rate(raw, 2)
    if name in RATE_1DEC:
        return parse_rate(raw, 1)
    if name in INT_FIELDS:
        return parse_int(value)
    return value or None


def parse_record(line: str) -> dict | None:
    raw = line.encode("cp932")
    if len(raw) < RECORD_SIZE:
        return None

    row: dict = {}
    era = ""
    birth_raw = ""
    pos = 0
    for name, size in FIELD_SPEC:
        chunk = raw[pos : pos + size].decode("cp932", errors="replace")
        if name == "年号":
            era = chunk.strip()
        elif name == "生年月日":
            birth_raw = chunk.strip()
        if name not in SKIP_FIELDS:
            row[name] = convert_field(name, chunk)
        pos += size

    row["生年月日_変換"] = parse_birth_date(era, birth_raw)
    return row


def file_sort_key(path: Path) -> tuple[int, str]:
    m = re.match(r"fan(\d{2})(\d{2})", path.stem, re.I)
    if not m:
        return (9999, path.name)
    return (int(m.group(1)), int(m.group(2)))


def extract_records(years: range) -> list[dict]:
    records: list[dict] = []

    for year in years:
        year_dir = INPUT_DIR / f"{year}年"
        print("processing:", year_dir)
        if not year_dir.exists():
            continue

        for txt_file in sorted(year_dir.glob("fan*.txt"), key=file_sort_key):
            with open(txt_file, encoding="cp932", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    row = parse_record(line)
                    if row:
                        records.append(row)

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n=== {dataset['output']} ===")
        records = extract_records(dataset["years"])
        df = pd.DataFrame(records)[OUTPUT_COLUMNS]

        output_path = OUTPUT_DIR / dataset["output"]
        df.to_csv(output_path, index=False, encoding="UTF-8-sig")

        print("総件数:", len(df))
        if len(df) > 0:
            print("登番数:", df["登番"].nunique())
            print("年×期:", df.groupby(["年", "期"]).size().to_dict())
            print("保存:", output_path)


if __name__ == "__main__":
    main()
