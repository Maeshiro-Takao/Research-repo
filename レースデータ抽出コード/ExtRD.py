"""
元データからの抽出を行う。

- グレード: 丸亀グレード別レース2014_2025.csv
- レース: 競走成績TXT（着・展示・気象・3連単オッズ）
- 選手: fan*.txt（登番・級・全国勝率など）

計算は CalcRD.py、統合出力は ComRD.py。

使い方:
    python レースデータ抽出コード/ExtRD.py   # グレードCSV生成
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GRADE_SOURCE_DIR = BASE_DIR / "元データ" / "丸亀グレード別レース名2014_2025"
RACE_INPUT_DIR = BASE_DIR / "元データ" / "競走成績"
OUTPUT_PATH = BASE_DIR / "元データ" / "丸亀グレード別レース2014_2025.csv"
MARUGAME_CODE = "15"

OUTPUT_COLUMNS = ["開催日", "日目", "レース", "グレード", "レース名", "優勝戦"]
RACE_KEY = ["開催日", "日目", "レース"]

RACE_HEADER = re.compile(r"^(\d+)R\s+.+H\d+m")
BLOCK_DATE = re.compile(r"第\s*(\d+)日\s+(\d{4})/\s*(\d+)/\s*(\d+)")


def normalize_race_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name)
    return re.sub(r"\s+", "", text)


def load_meeting_grade_lookup() -> dict[str, str]:
    """丸亀グレード別レース名 CSV からレース名→グレードの辞書を構築"""
    lookup: dict[str, str] = {}
    for csv_path in sorted(GRADE_SOURCE_DIR.glob("*/*_ALL.csv")):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = normalize_race_name(str(row["race_name"]))
            lookup[key] = str(row["grade"])
    return lookup


def resolve_grade(meeting_title: str | None, grade_lookup: dict[str, str]) -> str:
    if not meeting_title:
        return "一般"
    key = normalize_race_name(meeting_title)
    if key in grade_lookup:
        return grade_lookup[key]
    for name, grade in grade_lookup.items():
        if name in key or key in name:
            return grade
    return "一般"


def extract_meeting_title(block_lines: list[str]) -> str | None:
    for i, line in enumerate(block_lines):
        if line.startswith("＊") and "競走成績" in line:
            for candidate in block_lines[i + 1 :]:
                if not candidate:
                    continue
                if candidate.startswith("第") and "日" in candidate:
                    break
                return candidate
            break
    return None


def parse_block_date(
    block_lines: list[str],
    fallback_date: str | None = None,
) -> tuple[str | None, int | None]:
    for line in block_lines:
        m = BLOCK_DATE.search(line)
        if m:
            day_num = int(m.group(1))
            year = int(m.group(2))
            month = int(m.group(3))
            day = int(m.group(4))
            return f"{year}-{month:02d}-{day:02d}", day_num
    return fallback_date, None


def is_championship_race(race_header_line: str) -> bool:
    h_pos = race_header_line.find("H")
    if h_pos < 0:
        return False
    r_pos = race_header_line.find("R")
    if r_pos < 0:
        return False
    race_type = normalize_race_name(race_header_line[r_pos + 1 : h_pos])
    return "優勝戦" in race_type and "準優勝" not in race_type


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


def extract_grade_rows_from_block(
    block_lines: list[str],
    fallback_date: str | None,
    grade_lookup: dict[str, str],
) -> list[dict]:
    meeting_date, day_number = parse_block_date(block_lines, fallback_date)
    if meeting_date is None or day_number is None:
        return []

    meeting_title = extract_meeting_title(block_lines)
    grade = resolve_grade(meeting_title, grade_lookup)
    race_name = meeting_title or ""

    rows: list[dict] = []
    for line in block_lines:
        race_match = RACE_HEADER.match(line)
        if not race_match:
            continue
        rows.append(
            {
                "開催日": meeting_date,
                "日目": day_number,
                "レース": int(race_match.group(1)),
                "グレード": grade,
                "レース名": race_name,
                "優勝戦": is_championship_race(line),
            }
        )
    return rows


def build_grade_race_table(years: range | None = None) -> pd.DataFrame:
    """競走成績からグレード・優勝戦付きレース表を構築"""
    if years is None:
        years = range(14, 26)

    grade_lookup = load_meeting_grade_lookup()
    records: list[dict] = []

    for year in years:
        year_dir = RACE_INPUT_DIR / f"{year}年"
        if not year_dir.exists():
            continue

        for txt_file in sorted(year_dir.rglob("*.TXT"), key=file_sort_key):
            fallback_date = extract_date_from_filename(txt_file.name)
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
                        records.extend(
                            extract_grade_rows_from_block(
                                block_lines, fallback_date, grade_lookup
                            )
                        )
                    in_block = False
                    block_lines = []
                    continue
                if in_block:
                    block_lines.append(s)

    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=RACE_KEY, keep="last")
    df = df.sort_values(RACE_KEY).reset_index(drop=True)
    return df[OUTPUT_COLUMNS]


def load_grade_race_table(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or OUTPUT_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"グレードCSVが見つかりません: {csv_path}\n"
            "先に python レースデータ抽出コード/ExtRD.py を実行してください。"
        )
    df = pd.read_csv(csv_path)
    if "優勝戦" in df.columns:
        df["優勝戦"] = df["優勝戦"].map(_to_bool)
    return df


def _to_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def save_grade_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    output_path = path or OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="UTF-8-sig")
    return output_path


def attach_grade_info(
    race_df: pd.DataFrame,
    grade_df: pd.DataFrame,
) -> pd.DataFrame:
    """丸亀グレード別レースCSVからグレード・優勝戦をレース行へ付与"""
    grade_cols = grade_df.rename(
        columns={"グレード": "grade", "優勝戦": "is_championship"}
    )
    merged = race_df.merge(
        grade_cols[["開催日", "日目", "レース", "grade", "is_championship"]],
        on=["開催日", "日目", "レース"],
        how="left",
    )
    merged["grade"] = merged["grade"].fillna("一般")
    merged["is_championship"] = merged["is_championship"].fillna(False).astype(bool)
    return merged


# --- 選手データ抽出 (fan*.txt) ---

PLAYER_INPUT_DIR = BASE_DIR / "元データ" / "選手データ"

EXCLUDED_RESULT_CODES = ("F", "L0", "L1", "K0", "K1", "S0", "S1", "S2")
SKIP_FIELDS = {
    "名前カナ", "支部", "年号", "生年月日", "性別", "年齢", "血液型",
    "コースなしL0回数", "コースなしL1回数", "コースなしK0回数", "コースなしK1回数",
    "前期級", "前々期級", "前々々期級", "前期能力指数", "今期能力指数", "養成期", "出身地",
}
for _course in range(1, 7):
    for _code in EXCLUDED_RESULT_CODES:
        SKIP_FIELDS.add(f"{_course}コース{_code}回数")

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

PLAYER_DEDUP_KEY = ["登番", "年", "期"]


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


def parse_player_record(line: str) -> dict | None:
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


def fan_file_sort_key(path: Path) -> tuple[int, str]:
    m = re.match(r"fan(\d{2})(\d{2})", path.stem, re.I)
    if not m:
        return (9999, path.name)
    return (int(m.group(1)), int(m.group(2)))


def dedupe_player_records(records: list[dict]) -> list[dict]:
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


def extract_player_records(years: range) -> list[dict]:
    records: list[dict] = []

    for year in years:
        year_dir = PLAYER_INPUT_DIR / f"{year}年"
        if not year_dir.exists():
            continue

        for txt_file in sorted(year_dir.glob("fan*.txt"), key=fan_file_sort_key):
            source_key = fan_file_sort_key(txt_file)
            with open(txt_file, encoding="cp932", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    row = parse_player_record(line)
                    if row:
                        row["_source_key"] = source_key
                        records.append(row)

    return dedupe_player_records(records)


def extract_player_dataframe(years: range) -> pd.DataFrame:
    return pd.DataFrame(extract_player_records(years))


# --- レース結果抽出 (競走成績TXT) ---

INVALID_RESULT = re.compile(r"^(F|L|K\d*|S[012]|\.)$")
WEATHER_PATTERN = re.compile(
    r"(晴|曇り|雨|小雨|霧雨|雪|くもり|曇)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm"
)
PAYOUT_HEADER = "[払戻金]"
PAYOUT_RACE = re.compile(r"^(\d+)R\s+(\S+)\s+(\S+)")
RACE_DETAIL = re.compile(r"^\d+R\s+")


def parse_payout_line(line: str) -> tuple[int, str, float] | None:
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

    return race_no, combo_str, payout / 100.0


def parse_payout_section(block_lines: list[str]) -> dict[int, tuple[str, float]]:
    payouts: dict[int, tuple[str, float]] = {}
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
        payouts[race_no] = (combo_str, odds)

    return payouts


def safe_float(value: str) -> float | None:
    value = value.strip()
    if not value or value == ".":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_racer_row(parts, current_date, day_number, current_race, current_weather):
    chaku_raw = parts[0]
    if not chaku_raw.isdigit():
        return None

    chaku = int(chaku_raw)
    tei = int(parts[1])
    toban = int(parts[2])

    name_end = None
    for i in range(3, len(parts)):
        if parts[i].isdigit():
            name_end = i
            break

    if name_end is None:
        return None

    name = "".join(parts[3:name_end])
    remain = parts[name_end:]

    if len(remain) < 3:
        return None

    return {
        "開催日": current_date,
        "日目": day_number,
        "レース": current_race,
        "着": chaku,
        "艇": tei,
        "登番": toban,
        "選手名": name,
        "モーター": int(remain[0]),
        "ボート": int(remain[1]),
        "展示": safe_float(remain[2]),
        **current_weather,
    }


def finalize_race(
    race_rows: list[dict],
    current_weather,
    race_invalid: bool,
    current_date,
    day_number,
    current_race: int | None,
    payouts: dict[int, tuple[str, float]],
) -> list[dict]:
    if (
        race_invalid
        or current_weather is None
        or len(race_rows) != 6
        or day_number is None
        or current_date is None
        or current_race is None
    ):
        return []

    _, trifecta_odds = payouts.get(current_race, (None, None))
    finalized = []
    for row in race_rows:
        row = row.copy()
        row["3連単オッズ"] = trifecta_odds
        finalized.append(row)
    return finalized


def process_marugame_block(
    block_lines: list[str],
    fallback_date: str | None,
) -> list[dict]:
    meeting_date, day_number = parse_block_date(block_lines, fallback_date)
    if meeting_date is None or day_number is None:
        return []

    payouts = parse_payout_section(block_lines)

    block_records: list[dict] = []
    current_race = None
    current_weather = None
    race_rows: list[dict] = []
    race_invalid = False

    for line in block_lines:
        if not line:
            continue

        race_match = RACE_HEADER.match(line)
        if race_match:
            block_records.extend(
                finalize_race(
                    race_rows, current_weather, race_invalid,
                    meeting_date, day_number, current_race, payouts,
                )
            )

            current_race = int(race_match.group(1))
            race_rows = []
            race_invalid = False

            weather_match = WEATHER_PATTERN.search(line)
            if weather_match:
                current_weather = {
                    "天気": weather_match.group(1),
                    "風向": weather_match.group(2),
                    "風速": int(weather_match.group(3)),
                    "波高": int(weather_match.group(4)),
                }
            else:
                current_weather = None
            continue

        parts = line.split()
        if len(parts) < 8:
            continue

        if parts[0] not in {"01", "02", "03", "04", "05", "06"}:
            if INVALID_RESULT.fullmatch(parts[0]):
                race_invalid = True
            continue

        if current_race is None or current_weather is None:
            continue

        try:
            row = parse_racer_row(
                parts, meeting_date, day_number, current_race, current_weather
            )
            if row:
                race_rows.append(row)
        except (ValueError, IndexError):
            continue

    block_records.extend(
        finalize_race(
            race_rows, current_weather, race_invalid,
            meeting_date, day_number, current_race, payouts,
        )
    )
    return block_records


def extract_race_records(years: range) -> list[dict]:
    records: list[dict] = []

    for year in years:
        year_dir = RACE_INPUT_DIR / f"{year}年"
        if not year_dir.exists():
            continue

        for txt_file in sorted(year_dir.rglob("*.TXT"), key=file_sort_key):
            fallback_date = extract_date_from_filename(txt_file.name)
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
                        records.extend(
                            process_marugame_block(block_lines, fallback_date)
                        )
                    in_block = False
                    block_lines = []
                    continue
                if in_block:
                    block_lines.append(s)

    return records


def extract_race_dataframe(years: range) -> pd.DataFrame:
    return pd.DataFrame(extract_race_records(years))


def main():
    print("競走成績からグレード・優勝戦を抽出...")
    df = build_grade_race_table()
    output_path = save_grade_csv(df)

    print("総件数:", len(df))
    if len(df) > 0:
        print("開催日数:", df["開催日"].nunique())
        print("レース名数:", df["レース名"].nunique())
        print("グレード内訳:", df["グレード"].value_counts().to_dict())
        print("優勝戦:", int(df["優勝戦"].sum()), f"({df['優勝戦'].mean():.1%})")
    print("保存:", output_path)


if __name__ == "__main__":
    main()
