import re
from pathlib import Path
import pandas as pd

# ============================================================
# 基本設定
# ============================================================

base_dir = Path("/Users/maeshirotakao/ボートデータ/競走成績")
MARUGAME_CODE = "15"
OUTPUT_FILE = "丸亀学習用_レースデータ.csv"

INVALID_RESULT = re.compile(r"^(F|L|K\d*|S[012]|\.)$")
RACE_HEADER = re.compile(r"^(\d+)R\s+.+H\d+m")
DAY_PATTERN = re.compile(r"第\s*(\d+)日")

records = []

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


def parse_racer_row(parts, current_date, day_number, current_race):
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
    }


def finalize_race(race_rows, race_invalid, current_date, day_number):
    if (
        race_invalid
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
    race_rows = []
    race_invalid = False

    for line in block_lines:
        if not line:
            continue

        race_match = RACE_HEADER.match(line)
        if race_match:
            block_records.extend(
                finalize_race(race_rows, race_invalid, current_date, day_number)
            )

            current_race = int(race_match.group(1))
            race_rows = []
            race_invalid = False
            continue

        parts = line.split()
        if len(parts) < 8:
            continue

        if parts[0] not in {"01", "02", "03", "04", "05", "06"}:
            if INVALID_RESULT.fullmatch(parts[0]):
                race_invalid = True
            continue

        if current_race is None:
            continue

        try:
            row = parse_racer_row(parts, current_date, day_number, current_race)
            if row:
                race_rows.append(row)
        except (ValueError, IndexError):
            continue

    block_records.extend(
        finalize_race(race_rows, race_invalid, current_date, day_number)
    )
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
