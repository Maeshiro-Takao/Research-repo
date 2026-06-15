import re
from pathlib import Path
import pandas as pd

# ============================================================
# 基本設定
# ============================================================

base_dir = Path(__file__).resolve().parent.parent / "元データ" / "競走成績"
output_dir = Path(__file__).resolve().parent.parent / "編集データ"
MARUGAME_CODE = "15"

COLUMNS = [
    "開催日", "日目", "レース",
    "着", "艇",
    "登番", "選手名", "モーター", "ボート", "展示",
    "天気", "風向", "風速", "波高",
]

INVALID_RESULT = re.compile(r"^(F|L|K\d*|S[012]|\.)$")
RACE_HEADER = re.compile(r"^(\d+)R\s+.+H\d+m")
WEATHER_PATTERN = re.compile(
    r"(晴|曇り|雨|小雨|霧雨|雪)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm"
)
DAY_PATTERN = re.compile(r"第\s*(\d+)日")

DATASETS = [
    {"years": range(14, 25), "output": "丸亀学習用_レースデータ.csv"},
    {"years": range(25, 26), "output": "丸亀テスト用_レースデータ.csv"},
]

# ============================================================
# ユーティリティ
# ============================================================

def extract_date_from_filename(filename: str):
    m = re.match(r"K(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        return None

    yy, mm, dd = m.groups()
    year = 2000 + int(yy)
    return f"{year}-{int(mm):02d}-{int(dd):02d}"


def file_sort_key(path: Path):
    m = re.match(r"K(\d{2})(\d{2})(\d{2})", path.name)
    if not m:
        return (9999, 99, 99)

    yy, mm, dd = m.groups()
    return (int(yy), int(mm), int(dd))


def extract_day_number(block_lines):
    for line in block_lines:
        m = DAY_PATTERN.search(line)
        if m:
            return int(m.group(1))
    return None


def parse_racer_row(parts, current_date, day_number, current_race, current_weather):
    chaku = int(parts[0])
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
        "展示": float(remain[2]),
        **current_weather,
    }


def finalize_race(race_rows, current_weather, race_invalid, current_date, day_number):
    if (
        race_invalid
        or current_weather is None
        or len(race_rows) != 6
        or day_number is None
        or current_date is None
    ):
        return []

    return race_rows


def process_marugame_block(block_lines, current_date):
    day_number = extract_day_number(block_lines)
    if day_number is None:
        return []

    block_records = []
    current_race = None
    current_weather = None
    race_rows = []
    race_invalid = False

    for line in block_lines:
        if not line:
            continue

        race_match = RACE_HEADER.match(line)
        if race_match:
            block_records.extend(
                finalize_race(race_rows, current_weather, race_invalid, current_date, day_number)
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
                parts, current_date, day_number, current_race, current_weather
            )
            if row:
                race_rows.append(row)
        except (ValueError, IndexError):
            continue

    block_records.extend(
        finalize_race(race_rows, current_weather, race_invalid, current_date, day_number)
    )
    return block_records


def extract_records(years: range) -> list:
    records = []

    for year in years:
        year_dir = base_dir / f"{year}年"
        print("processing:", year_dir)

        if not year_dir.exists():
            continue

        for txt_file in sorted(year_dir.rglob("*.TXT"), key=file_sort_key):
            current_date = extract_date_from_filename(txt_file.name)

            with open(txt_file, encoding="UTF-8") as f:
                lines = f.readlines()

            in_block = False
            block_lines = []

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


# ============================================================
# メイン
# ============================================================

output_dir.mkdir(parents=True, exist_ok=True)

for dataset in DATASETS:
    print(f"\n=== {dataset['output']} ===")
    records = extract_records(dataset["years"])

    df = pd.DataFrame(records, columns=COLUMNS)
    output_path = output_dir / dataset["output"]
    df.to_csv(output_path, index=False, encoding="UTF-8-sig")

    print("総件数:", len(df))
    if len(df) > 0:
        print("開催日数:", df["開催日"].nunique())
        print("保存:", output_path)
