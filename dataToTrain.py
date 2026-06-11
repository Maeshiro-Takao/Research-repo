import re
from pathlib import Path
import pandas as pd

INPUT_DIR = Path("/Users/maeshirotakao/ボートデータ/競走成績")
OUTPUT_CSV = Path("/Users/maeshirotakao/ボートデータ/丸亀データ_学習用.csv")

ABNORMAL_CODES = {"F", "L0", "L1", "S0", "S1", "S2", "K0", "K1"}


def parse_date_from_filename(path: Path):
    m = re.fullmatch(r"K(\d{6})\.[Tt][Xx][Tt]", path.name)
    if not m:
        return None

    yymmdd = m.group(1)
    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])

    return f"{2000 + yy:04d}-{mm:02d}-{dd:02d}"


def read_text(path: Path):
    for enc in ["cp932", "shift_jis", "utf-8", "latin-1"]:
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            pass
    return ""


def extract_marugame_block(text: str):
    bgn = re.search(r"^\s*15KBGN\s*$", text, flags=re.MULTILINE)
    if not bgn:
        return None

    end = re.search(r"^\s*15KEND\s*$", text[bgn.end():], flags=re.MULTILINE)
    if not end:
        return None

    start_pos = bgn.end()
    end_pos = start_pos + end.start()
    block = text[start_pos:end_pos]

    first_race = re.search(r"^\s*1R\s+", block, flags=re.MULTILINE)
    if not first_race:
        return None

    return block[first_race.start():]


def to_int(s):
    s = s.strip()
    return int(s) if re.fullmatch(r"\d+", s) else None


def to_float(s):
    s = s.strip()
    return float(s) if re.fullmatch(r"\d+\.\d+", s) else None


def is_racetime_token(s):
    return re.fullmatch(r"\d+\.\d+\.\d+", s.strip()) is not None


def parse_start_token(s):
    s = s.strip()

    if re.fullmatch(r"\d+\.\d+", s):
        return float(s)

    m = re.fullmatch(r"[FL](\d+\.\d+)", s)
    if m:
        return float(m.group(1))

    return None


def parse_one_race(race_text: str, date_str: str, race_no: int):
    rows = []

    for line in race_text.splitlines():
        line = line.rstrip()

        # 実ファイルは「着 艇 登番 ...」
        if not re.match(r"^\s*(\d{2}|F|L0|L1|K0|K1|S0|S1|S2)\s+\d\s+\d{4}\s+", line):
            continue

        toks = line.split()
        if len(toks) < 8:
            continue

        chaku_raw = toks[0]

        # 異常艇が1つでもあればそのレース全体を除外
        if chaku_raw in ABNORMAL_CODES:
            return []

        # 末尾のレースタイムは不要なので削る
        if is_racetime_token(toks[-1]):
            toks = toks[:-1]

        if len(toks) < 8:
            continue

        # 実データの列順は 着, 艇, 登番
        chaku = to_int(toks[0])
        tei = to_int(toks[1])
        touban = to_int(toks[2])

        if chaku is None or tei is None or touban is None:
            continue

        tenji = to_float(toks[-3])
        shinnyu = to_int(toks[-2])
        start = parse_start_token(toks[-1])

        # 正常レースなら tenji と shinnyu は数値で取れる
        if tenji is None or shinnyu is None:
            return []

        motor = None
        if len(toks) >= 9 and re.fullmatch(r"\d+", toks[-5]):
            motor = int(toks[-5])
        elif len(toks) >= 8 and re.fullmatch(r"\d+", toks[-4]):
            motor = int(toks[-4])

        rows.append({
            "開催日": date_str,
            "レース": race_no,
            "着": chaku,
            "艇": tei,
            "登番": touban,
            "モーター": motor,
            "展示": tenji,
            "進入": shinnyu,
            "スタート": start,
        })

    # 正常レースなら通常6艇分ある想定
    if len(rows) != 6:
        return []

    return rows


def parse_rows_from_block(block: str, date_str: str):
    rows = []
    headers = list(re.finditer(r"^\s*(\d{1,2})R\s+", block, flags=re.MULTILINE))

    for i, m in enumerate(headers):
        race_no = int(m.group(1))
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(block)
        race_text = block[start:end]

        race_rows = parse_one_race(race_text, date_str, race_no)
        rows.extend(race_rows)

    return rows


def main():
    all_rows = []

    files = sorted(INPUT_DIR.rglob("K*.[Tt][Xx][Tt]"))
    files = [p for p in files if re.fullmatch(r"K\d{6}\.[Tt][Xx][Tt]", p.name)]

    for fp in files:
        date_str = parse_date_from_filename(fp)
        if date_str is None:
            continue

        text = read_text(fp)
        if not text:
            continue

        block = extract_marugame_block(text)
        if block is None:
            continue

        rows = parse_rows_from_block(block, date_str)
        all_rows.extend(rows)

    if not all_rows:
        print("抽出できたデータがありません。")
        return

    df = pd.DataFrame(all_rows)

    df["開催日"] = pd.to_datetime(df["開催日"], errors="coerce")
    df = df.sort_values(["開催日", "レース", "艇"]).reset_index(drop=True)
    df["開催日"] = df["開催日"].dt.strftime("%Y-%m-%d")

    # ここで着と艇を強制修正
    df[["着", "艇"]] = df[["艇", "着"]].to_numpy()

    df = df[["開催日", "レース", "着", "艇", "登番", "モーター", "展示", "進入", "スタート"]]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"保存完了: {OUTPUT_CSV}")
    print(df.head(20))


if __name__ == "__main__":
    main()