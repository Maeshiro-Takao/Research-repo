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
    "直近10年勝率", "直近10年2連帯率", "直近10年3連帯率",
    "直近5年勝率", "直近5年2連帯率", "直近5年3連帯率",
    "1着率", "2着率", "3着率",
    "1着回数", "2着回数", "出走回数", "優出回数", "優勝回数", "平均スタートタイミング",
]
for _course in range(1, 7):
    OUTPUT_COLUMNS += [
        f"{_course}コース進入回数",
        f"{_course}コース複勝率",
        f"{_course}コース平均スタートタイミング",
        f"{_course}コース平均スタート順位",
    ]
for _course in range(1, 7):
    for _rank in range(1, 7):
        OUTPUT_COLUMNS.append(f"{_course}コース{_rank}着回数")
OUTPUT_COLUMNS += ["算出期間自", "算出期間至"]

ROLLING_WINDOWS = (10, 5)
RANK3_COLUMNS = [f"{c}コース3着回数" for c in range(1, 7)]
ROLLING_RATE_COLUMNS = [
    "直近10年勝率", "直近10年2連帯率", "直近10年3連帯率",
    "直近5年勝率", "直近5年2連帯率", "直近5年3連帯率",
]
DEDUP_KEY = ["登番", "年", "期"]

EXCLUDED_RESULT_CODES = ("F", "L0", "L1", "K0", "K1", "S0", "S1", "S2")
SKIP_FIELDS = {
    "名前カナ", "支部", "年号", "生年月日", "性別", "年齢", "血液型",
    "コースなしL0回数", "コースなしL1回数", "コースなしK0回数", "コースなしK1回数",
    "前期級", "前々期級", "前々々期級", "前期能力指数", "今期能力指数", "養成期", "出身地",
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

RATE_2DEC = {"勝率", "平均スタートタイミング"}
RATE_2DEC |= {f"{c}コース平均スタートタイミング" for c in range(1, 7)}
RATE_2DEC |= {f"{c}コース平均スタート順位" for c in range(1, 7)}

RATE_1DEC = {"複勝率"}
RATE_1DEC |= {f"{c}コース複勝率" for c in range(1, 7)}

INT_FIELDS = {
    "登番", "年齢", "身長", "体重", "1着回数", "2着回数", "出走回数",
    "優出回数", "優勝回数", "年",
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


def parse_yyyymmdd(value: str) -> str | None:
    value = value.strip()
    if len(value) != 8 or not value.isdigit():
        return None
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def convert_field(name: str, raw: str) -> str | int | float | None:
    value = raw.strip()
    if name in {"名前漢字", "名前カナ", "支部", "級", "年号", "血液型"}:
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
    pos = 0
    for name, size in FIELD_SPEC:
        chunk = raw[pos : pos + size].decode("cp932", errors="replace")
        if name not in SKIP_FIELDS:
            row[name] = convert_field(name, chunk)
        pos += size

    return row


def file_sort_key(path: Path) -> tuple[int, str]:
    m = re.match(r"fan(\d{2})(\d{2})", path.stem, re.I)
    if not m:
        return (9999, path.name)
    return (int(m.group(1)), int(m.group(2)))


def dedupe_records(records: list[dict]) -> list[dict]:
    """登番×年×期の重複を除去し、最新ファイルのレコードを残す"""
    best: dict[tuple, dict] = {}
    for row in records:
        key = (row.get("登番"), row.get("年"), row.get("期"))
        if key[0] is None:
            continue
        source_key = row.pop("_source_key", (0, ""))
        existing = best.get(key)
        if existing is None or source_key > existing["_source_key"]:
            row["_source_key"] = source_key
            best[key] = row

    deduped = []
    for row in best.values():
        row.pop("_source_key", None)
        deduped.append(row)
    return deduped


def round_rate(value: float | None, decimals: int) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), decimals)


def add_third_place_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_3着回数"] = df[RANK3_COLUMNS].fillna(0).sum(axis=1).astype(int)
    return df


def compute_rolling_rates(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """算出期間至を基準に直近10年/5年の勝率・2連帯率・3連帯率を付与"""
    target = add_third_place_count(df)
    if history_df is not None:
        pool = pd.concat([add_third_place_count(history_df), target], ignore_index=True)
    else:
        pool = target.copy()
    pool = pool.drop_duplicates(subset=DEDUP_KEY, keep="last")
    pool["算出期間至"] = pd.to_datetime(pool["算出期間至"])
    pool = pool.sort_values(["登番", "算出期間至"]).reset_index(drop=True)

    rate_values: dict[tuple, dict[str, float | None]] = {}
    for _, group in pool.groupby("登番", sort=False):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            ref_date = row["算出期間至"]
            key = (row["登番"], row["年"], row["期"])
            rates: dict[str, float | None] = {}

            for years in ROLLING_WINDOWS:
                start_date = ref_date - pd.DateOffset(years=years)
                hist = group[
                    (group["算出期間至"] <= ref_date)
                    & (group["算出期間至"] > start_date)
                ]
                total_runs = hist["出走回数"].fillna(0).sum()
                prefix = f"直近{years}年"

                if total_runs <= 0:
                    rates[f"{prefix}勝率"] = None
                    rates[f"{prefix}2連帯率"] = None
                    rates[f"{prefix}3連帯率"] = None
                    continue

                win_points = (hist["勝率"].fillna(0) * hist["出走回数"].fillna(0)).sum()
                first = hist["1着回数"].fillna(0).sum()
                second = hist["2着回数"].fillna(0).sum()
                third = hist["_3着回数"].fillna(0).sum()

                rates[f"{prefix}勝率"] = round_rate(win_points / total_runs, 2)
                rates[f"{prefix}2連帯率"] = round_rate((first + second) / total_runs * 100, 1)
                rates[f"{prefix}3連帯率"] = round_rate((first + second + third) / total_runs * 100, 1)

            rate_values[key] = rates

    for col in ROLLING_RATE_COLUMNS:
        target[col] = [
            rate_values.get((row["登番"], row["年"], row["期"]), {}).get(col)
            for _, row in target.iterrows()
        ]

    return target.drop(columns=["_3着回数"])


def add_place_rates(df: pd.DataFrame) -> pd.DataFrame:
    """1着率・2着率・3着率を追加する"""
    df = df.copy()
    third = df[RANK3_COLUMNS].fillna(0).sum(axis=1)
    runs = df["出走回数"].fillna(0)

    for name, counts in (
        ("1着率", df["1着回数"].fillna(0)),
        ("2着率", df["2着回数"].fillna(0)),
        ("3着率", third),
    ):
        df[name] = [
            round_rate(c / run * 100, 1) if run > 0 else None
            for c, run in zip(counts, runs)
        ]
    return df


def records_to_dataframe(
    records: list[dict],
    history_records: list[dict] | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(records)
    history_df = pd.DataFrame(history_records) if history_records else None
    df = compute_rolling_rates(df, history_df)
    df = add_place_rates(df)
    return df[OUTPUT_COLUMNS]


def extract_records(years: range) -> list[dict]:
    records: list[dict] = []

    for year in years:
        year_dir = INPUT_DIR / f"{year}年"
        print("processing:", year_dir)
        if not year_dir.exists():
            continue

        for txt_file in sorted(year_dir.glob("fan*.txt"), key=file_sort_key):
            source_key = file_sort_key(txt_file)
            with open(txt_file, encoding="cp932", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    row = parse_record(line)
                    if row:
                        row["_source_key"] = source_key
                        records.append(row)

    return dedupe_records(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history_records: list[dict] | None = None
    for dataset in DATASETS:
        print(f"\n=== {dataset['output']} ===")
        records = extract_records(dataset["years"])
        df = records_to_dataframe(records, history_records)

        output_path = OUTPUT_DIR / dataset["output"]
        df.to_csv(output_path, index=False, encoding="UTF-8-sig")

        print("総件数:", len(df))
        if len(df) > 0:
            meta = pd.DataFrame(records)
            print("登番数:", df["登番"].nunique())
            print("年×期:", meta.groupby(["年", "期"]).size().to_dict())
            dup_count = meta.duplicated(DEDUP_KEY).sum()
            print("重複件数:", dup_count)
            print("保存:", output_path)

        if dataset["output"] == "丸亀学習用_選手データ.csv":
            history_records = records


if __name__ == "__main__":
    main()
