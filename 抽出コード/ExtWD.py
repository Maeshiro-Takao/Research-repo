import re
from pathlib import Path
import pandas as pd

# ============================================================
# 基本設定
# ============================================================

base_dir = Path("/Users/maeshirotakao/ボートデータ/競走成績")
MARUGAME_CODE = "15"
OUTPUT_FILE = "丸亀学習用_気象データ.csv"

INVALID_RESULT = re.compile(r"^(F|L|K\d*|S[012]|\.)$")
WEATHER_PATTERN = re.compile(
    r"(晴|曇り|雨|小雨|霧雨|雪)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm"
)
DAY_PATTERN = re.compile(r"第\s*(\d+)日")

records = []

# ============================================================
# ファイル名日付
# ============================================================

def extract_date_from_filename(filename: str):
    m = re.match(r"K(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        return None

    yy, mm, dd = m.groups()
    year = 2000 + int(yy)

    return f"{year}-{int(mm):02d}-{int(dd):02d}"

# ============================================================
# ファイルソート
# ============================================================

def file_sort_key(path: Path):
    m = re.match(r"K(\d{2})(\d{2})(\d{2})", path.name)
    if not m:
        return (9999, 99, 99)

    yy, mm, dd = m.groups()
    return (int(yy), int(mm), int(dd))

# ============================================================
# 丸亀ブロック解析
# ============================================================

def extract_day_number(block_lines):
    for line in block_lines:
        m = DAY_PATTERN.search(line)
        if m:
            return int(m.group(1))
    return None

def is_invalid_result(parts):
    return bool(INVALID_RESULT.fullmatch(parts[0]))

def finalize_race(current_race, current_weather, valid_boat_count, race_invalid, current_date, day_number):
    if (
        current_race is None
        or race_invalid
        or current_weather is None
        or valid_boat_count != 6
        or day_number is None
        or current_date is None
    ):
        return None

    return {
        "開催日": current_date,
        "日目": day_number,
        "レース": current_race,
        **current_weather,
    }

def process_marugame_block(block_lines, current_date):
    day_number = extract_day_number(block_lines)
    if day_number is None:
        return []

    block_records = []
    current_race = None
    current_weather = None
    valid_boat_count = 0
    race_invalid = False

    for line in block_lines:
        if not line:
            continue

        race_match = re.match(r"^(\d+)R\s", line)
        if race_match:
            record = finalize_race(
                current_race,
                current_weather,
                valid_boat_count,
                race_invalid,
                current_date,
                day_number,
            )
            if record:
                block_records.append(record)

            current_race = int(race_match.group(1))
            valid_boat_count = 0
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
            if is_invalid_result(parts):
                race_invalid = True
            continue

        if current_race is not None:
            valid_boat_count += 1

    record = finalize_race(
        current_race,
        current_weather,
        valid_boat_count,
        race_invalid,
        current_date,
        day_number,
    )
    if record:
        block_records.append(record)

    return block_records

# ============================================================
# メイン
# ============================================================

for year in range(14, 25):
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

# ============================================================
# 出力
# ============================================================

df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False, encoding="UTF-8-sig")

print("総件数:", len(df))

if len(df) > 0:
    print("開催日数:", df["開催日"].nunique())
    print(df["開催日"].value_counts().head(20))
