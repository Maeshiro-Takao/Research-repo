"""
競艇オッズ保管庫 (orangebuoy.net) から丸亀競艇場の3連単オッズ履歴を取得する。

取得項目: 開催日, レース, 組み合わせ(120通り),
  締切時 / 電話投票2分前 / 5分前 / 10分前 オッズ

負荷軽減のため、2019〜2025 は年単位で分割取得する（1年≒1日）。

使い方（推奨: 1日1年ずつ）:
    python オッズ抽出コード/ScrapeOrangeBuoyOdds.py --year 2019
    python オッズ抽出コード/ScrapeOrangeBuoyOdds.py --year 2020
    ...
    python オッズ抽出コード/ScrapeOrangeBuoyOdds.py --year 2025

1年の途中で区切る場合:
    python オッズ抽出コード/ScrapeOrangeBuoyOdds.py --year 2019 --max-races 500

robots.txt: /wp/wp-admin/ のみ Disallow。/odds/ は許可。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "オッズデータ"
RACE_DATA_DIR = BASE_DIR / "レースデータ"
OUTPUT_CSV = OUTPUT_DIR / "丸亀オッズデータby保管庫_2019_2025.csv"
PROGRESS_JSON = OUTPUT_DIR / ".scrape_orangebuoy_progress.json"
LOG_FILE = OUTPUT_DIR / "scrape_orangebuoy.log"
FAILED_LOG = OUTPUT_DIR / "scrape_orangebuoy_failed.log"

BASE_URL = "http://www.orangebuoy.net/odds/"
STADIUM_CODE = 15
STADIUM_NAME = "丸亀"
ALL_YEARS = list(range(2019, 2026))

# オッズ時点: サイト mode パラメータ（20分前は取得しない）
MODE_MAP: dict[str, int] = {
    "締切時オッズ": 99,
    "2分前オッズ": 1,
    "5分前オッズ": 2,
    "10分前オッズ": 3,
}
ODDS_COLUMNS = ["10分前オッズ", "5分前オッズ", "2分前オッズ", "締切時オッズ"]
CSV_COLUMNS = ["開催日", "レース場", "レース", "組み合わせ", *ODDS_COLUMNS]

# 旧英語列名 → 日本語（既存CSV移行用）
COLUMN_RENAME = {
    "date": "開催日",
    "stadium": "レース場",
    "race_no": "レース",
    "combination": "組み合わせ",
    "odds_10m": "10分前オッズ",
    "odds_5m": "5分前オッズ",
    "odds_2m": "2分前オッズ",
    "odds_final": "締切時オッズ",
    "odds_20m": "20分前オッズ",
}

MIN_DELAY_SEC = 5.0
MAX_DELAY_SEC = 10.0
RACE_PAUSE_SEC = 3.0
MAX_RETRIES = 3
RETRY_WAIT_SEC = 15.0
FLUSH_EVERY_RACES = 5

RATE_LIMIT_MARKERS = ("アクセスが多いため", "半日ほど時間を空けて")


class SiteRateLimitError(Exception):
    """サイト側のアクセス制限"""

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def is_rate_limited(text: str) -> bool:
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def probe_site_access() -> bool:
    """開始前にサイトへ1回だけアクセスし、制限中でないか確認"""
    url = f"{BASE_URL}?day=1&month=1&year=2024&jyo={STADIUM_CODE}&r=1&mode=3"
    try:
        resp = requests.get(url, headers=SESSION_HEADERS, timeout=30)
        resp.encoding = resp.apparent_encoding or "utf-8"
        if is_rate_limited(resp.text):
            logging.error(
                "サイトがアクセス制限中です: %s",
                resp.text.strip()[:80],
            )
            return False
        if len(resp.text) < 1000:
            logging.warning("応答が短いです（%d bytes）。制限の可能性があります。", len(resp.text))
            return False
        return True
    except requests.RequestException as exc:
        logging.error("接続確認失敗: %s", exc)
        return False


def is_boat(text: str) -> bool:
    return len(text) == 1 and text in "123456"


def is_odds(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def all_trifecta_combos() -> set[tuple[int, int, int]]:
    return {
        (i, j, k)
        for i in range(1, 7)
        for j in range(1, 7)
        if j != i
        for k in range(1, 7)
        if k not in (i, j)
    }


def parse_trifecta_table(table) -> dict[tuple[int, int, int], float]:
    combos: dict[tuple[int, int, int], float] = {}
    second_boats: list[int | None] = [None] * 6

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        if sum(is_odds(td.get_text(strip=True)) for td in tds) < 3:
            continue

        i = 0
        group = 0
        while i < len(tds) and group < 6:
            td = tds[i]
            text = td.get_text(strip=True)
            rowspan = int(td.get("rowspan", 1))

            if (
                is_boat(text)
                and rowspan > 1
                and i + 2 < len(tds)
                and is_boat(tds[i + 1].get_text(strip=True))
                and is_odds(tds[i + 2].get_text(strip=True))
            ):
                second_boats[group] = int(text)
                third = int(tds[i + 1].get_text(strip=True))
                odds = float(tds[i + 2].get_text(strip=True))
                first = group + 1
                combos[(first, second_boats[group], third)] = odds
                i += 3
                group += 1
            elif is_boat(text) and i + 1 < len(tds) and is_odds(tds[i + 1].get_text(strip=True)):
                if second_boats[group] is None:
                    break
                third = int(text)
                odds = float(tds[i + 1].get_text(strip=True))
                first = group + 1
                combos[(first, second_boats[group], third)] = odds
                i += 2
                group += 1
            else:
                i += 1

    return combos


def find_trifecta_odds(soup: BeautifulSoup) -> dict[tuple[int, int, int], float]:
    best: dict[tuple[int, int, int], float] = {}
    for table in soup.find_all("table"):
        parsed = parse_trifecta_table(table)
        if len(parsed) > len(best):
            best = parsed
    return best


def combo_str(first: int, second: int, third: int) -> str:
    return f"{first}-{second}-{third}"


def build_race_url(date_str: str, race_no: int, mode: int) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (
        f"{BASE_URL}?day={dt.day}&month={dt.month}&year={dt.year}"
        f"&jyo={STADIUM_CODE}&r={race_no}&mode={mode}"
    )


def migrate_csv_if_needed() -> None:
    """旧形式（英語列名・20分前列）を日本語列名へ変換"""
    if not OUTPUT_CSV.exists():
        return
    df = pd.read_csv(OUTPUT_CSV)
    original_cols = list(df.columns)
    renamed = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    if renamed:
        df = df.rename(columns=renamed)
    drop_cols = [c for c in df.columns if c not in CSV_COLUMNS]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if list(df.columns) != CSV_COLUMNS:
        df = df.reindex(columns=CSV_COLUMNS)
    if list(df.columns) == original_cols and not drop_cols:
        return
    logging.info("CSV列名を日本語へ移行 (%s → %s)", original_cols, list(df.columns))
    df.to_csv(OUTPUT_CSV, index=False, encoding="UTF-8-sig")
    logging.info("CSV移行完了: %s", OUTPUT_CSV)


class OrangeBuoyScraper:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(SESSION_HEADERS)
        self.completed_races: set[str] = set()
        self.pending_rows: list[dict] = []
        self.stats = {
            "races_ok": 0,
            "races_failed": 0,
            "rows_written": 0,
            "rows_missing_odds": 0,
            "skipped": 0,
        }

    def load_progress(self) -> None:
        if PROGRESS_JSON.exists():
            with open(PROGRESS_JSON, encoding="utf-8") as f:
                data = json.load(f)
            self.completed_races = set(data.get("completed_races", []))
            logging.info("再開: 取得済みレース %d 件", len(self.completed_races))

        if OUTPUT_CSV.exists():
            try:
                existing = pd.read_csv(OUTPUT_CSV, usecols=["開催日", "レース"])
                for date_str, race_no in existing.drop_duplicates().itertuples(index=False):
                    self.completed_races.add(f"{date_str}|{int(race_no)}")
            except Exception as exc:
                logging.warning("既存CSV読み込み失敗: %s", exc)

    def save_progress(self) -> None:
        with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
            json.dump({"completed_races": sorted(self.completed_races)}, f, ensure_ascii=False, indent=2)

    def _sleep(self) -> None:
        time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))

    def fetch_page(self, url: str) -> BeautifulSoup | None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or "utf-8"
                if is_rate_limited(resp.text):
                    raise SiteRateLimitError(resp.text.strip())
                return BeautifulSoup(resp.text, "html.parser")
            except SiteRateLimitError:
                raise
            except requests.RequestException as exc:
                logging.warning("取得失敗 (%d/%d): %s — %s", attempt, MAX_RETRIES, url, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SEC * attempt)
        return None

    def fetch_mode_odds(self, date_str: str, race_no: int, mode: int) -> dict[tuple[int, int, int], float]:
        url = build_race_url(date_str, race_no, mode)
        self._sleep()
        try:
            soup = self.fetch_page(url)
        except SiteRateLimitError:
            raise
        if soup is None:
            return {}
        if "オッズデータがありません" in soup.get_text():
            return {}

        odds_map = find_trifecta_odds(soup)
        if len(odds_map) < 100:
            logging.warning(
                "オッズ不足 (%d件): %s %dR mode=%d",
                len(odds_map), date_str, race_no, mode,
            )
        return odds_map

    def fetch_race_odds(
        self, date_str: str, race_no: int,
    ) -> tuple[dict[tuple[int, int, int], dict[str, float | None]] | None, list[str]]:
        merged: dict[tuple[int, int, int], dict[str, float | None]] = {}
        missing_modes: list[str] = []

        for col, mode in MODE_MAP.items():
            try:
                odds_map = self.fetch_mode_odds(date_str, race_no, mode)
            except SiteRateLimitError:
                raise
            if not odds_map:
                missing_modes.append(col)
                continue
            for combo, odds in odds_map.items():
                merged.setdefault(combo, {c: None for c in ODDS_COLUMNS})
                merged[combo][col] = odds

        if not merged:
            return None, missing_modes

        for combo in all_trifecta_combos():
            merged.setdefault(combo, {c: None for c in ODDS_COLUMNS})

        return merged, missing_modes

    def race_key(self, date_str: str, race_no: int) -> str:
        return f"{date_str}|{race_no}"

    def append_rows(self, rows: list[dict], *, write_header: bool = False) -> None:
        if not rows:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        mode = "w" if write_header else "a"
        with open(OUTPUT_CSV, mode, newline="", encoding="UTF-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        self.stats["rows_written"] += len(rows)

    def flush_pending(self, *, write_header: bool = False) -> None:
        if self.pending_rows:
            self.append_rows(self.pending_rows, write_header=write_header)
            self.pending_rows.clear()

    def log_failed(self, date_str: str, race_no: int, reason: str) -> None:
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(f"{date_str},{race_no},{reason}\n")

    def process_race(self, date_str: str, race_no: int) -> bool | None:
        """True=成功, False=データなし, None=アクセス制限で中断"""
        key = self.race_key(date_str, race_no)
        if key in self.completed_races:
            self.stats["skipped"] += 1
            return True

        try:
            odds_by_combo, missing_modes = self.fetch_race_odds(date_str, race_no)
        except SiteRateLimitError as exc:
            logging.error("アクセス制限を検出: %s", str(exc)[:80])
            return None
        if odds_by_combo is None:
            self.stats["races_failed"] += 1
            self.log_failed(date_str, race_no, "全モード取得失敗")
            logging.error("取得失敗: %s %dR", date_str, race_no)
            return False

        if missing_modes:
            logging.warning("部分欠損 %s %dR: %s", date_str, race_no, ", ".join(missing_modes))
            self.log_failed(date_str, race_no, f"部分欠損: {','.join(missing_modes)}")

        rows: list[dict] = []
        missing_cells = 0
        for combo in sorted(all_trifecta_combos()):
            values = odds_by_combo[combo]
            missing_cells += sum(v is None for v in values.values())
            rows.append({
                "開催日": date_str,
                "レース場": STADIUM_NAME,
                "レース": race_no,
                "組み合わせ": combo_str(*combo),
                **values,
            })

        self.pending_rows.extend(rows)
        self.stats["rows_missing_odds"] += missing_cells
        self.stats["races_ok"] += 1
        self.completed_races.add(key)
        time.sleep(RACE_PAUSE_SEC)
        return True

    def finalize_race_batch(self, *, first_write: bool) -> None:
        self.flush_pending(write_header=first_write)
        self.save_progress()


def load_race_schedule(years: range) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in years:
        path = RACE_DATA_DIR / f"丸亀_{year}_レースデータ.csv"
        if not path.exists():
            logging.warning("レースデータなし: %s", path)
            continue
        df = pd.read_csv(path, usecols=["開催日", "レース"])
        parts.append(df.drop_duplicates())

    if not parts:
        raise FileNotFoundError("レースデータCSVが見つかりません")

    return pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(
        ["開催日", "レース"],
    ).reset_index(drop=True)


def next_year_to_run(completed: set[str], years: list[int]) -> int | None:
    for year in years:
        path = RACE_DATA_DIR / f"丸亀_{year}_レースデータ.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["開催日", "レース"]).drop_duplicates()
        total = len(df)
        done = sum(1 for k in completed if k.startswith(f"{year}-"))
        if done < total:
            return year
    return None


def print_summary(scraper: OrangeBuoyScraper, total_races: int, target_years: list[int]) -> None:
    s = scraper.stats
    print("\n=== 取得サマリ ===")
    print(f"  対象年: {target_years[0]}〜{target_years[-1]}" if len(target_years) > 1 else f"  対象年: {target_years[0]}")
    print(f"  今回の対象レース数: {total_races}")
    print(f"  取得成功レース数: {s['races_ok']}")
    print(f"  取得失敗レース数: {s['races_failed']}")
    print(f"  スキップ（取得済み）: {s['skipped']}")
    print(f"  今回保存行数: {s['rows_written']}")
    print(f"  欠損オッズセル数: {s['rows_missing_odds']}")
    print(f"  出力: {OUTPUT_CSV}")

    nxt = next_year_to_run(scraper.completed_races, ALL_YEARS)
    if nxt is not None:
        print(f"\n  次回実行（推奨）:")
        print(f"    python オッズ抽出コード/ScrapeOrangeBuoyOdds.py --year {nxt}")
    else:
        print("\n  全期間（2019〜2025）の取得が完了しました。")


def main() -> None:
    parser = argparse.ArgumentParser(description="競艇オッズ保管庫スクレイパー（丸亀・年次分割）")
    parser.add_argument("--year", type=int, help="取得する年（1日1年ずつ推奨）")
    parser.add_argument("--years", nargs="+", type=int, help="複数年を指定（非推奨: 負荷大）")
    parser.add_argument("--max-races", type=int, help="今回取得する最大レース数（年の途中で区切る）")
    parser.add_argument("--test", action="store_true", help="1開催日・1Rのみ試行")
    parser.add_argument("--date", type=str, help="特定日のみ (YYYY-MM-DD)")
    parser.add_argument("--probe-only", action="store_true", help="接続確認のみ（取得しない）")
    args = parser.parse_args()

    setup_logging()
    migrate_csv_if_needed()

    if not probe_site_access():
        print("\nサイトがアクセス制限中です。半日〜24時間空けてから再実行してください。")
        print("  python オッズ抽出コード/ScrapeOrangeBuoyOdds.py --probe-only  # 確認のみ")
        sys.exit(2)
    if args.probe_only:
        print("接続OK — スクレイピング可能です。")
        return

    if args.year:
        target_years = [args.year]
    elif args.years:
        target_years = sorted(args.years)
        logging.warning("複数年一括取得はサイト負荷が高いです。--year で1年ずつ実行を推奨します。")
    else:
        nxt = None
        if PROGRESS_JSON.exists():
            with open(PROGRESS_JSON, encoding="utf-8") as f:
                completed = set(json.load(f).get("completed_races", []))
            nxt = next_year_to_run(completed, ALL_YEARS)
        target_years = [nxt if nxt is not None else ALL_YEARS[0]]
        logging.info("--year 未指定: 未完了の %d 年を自動選択", target_years[0])

    logging.info("競艇オッズ保管庫スクレイピング開始（丸亀 %s）", target_years)
    logging.info(
        "取得時点: 締切時 / 2分前 / 5分前 / 10分前 | アクセス間隔 %.1f〜%.1f 秒",
        MIN_DELAY_SEC, MAX_DELAY_SEC,
    )

    scraper = OrangeBuoyScraper()
    scraper.load_progress()

    years = range(min(target_years), max(target_years) + 1)
    schedule = load_race_schedule(years)

    if args.date:
        schedule = schedule.loc[schedule["開催日"] == args.date]
    if args.test:
        first_date = schedule["開催日"].iloc[0]
        schedule = schedule.loc[(schedule["開催日"] == first_date) & (schedule["レース"] == 1)]
    if args.max_races:
        pending = schedule.loc[
            ~schedule.apply(lambda r: scraper.race_key(str(r["開催日"]), int(r["レース"])) in scraper.completed_races, axis=1)
        ]
        schedule = pending.head(args.max_races)

    total_races = len(schedule)
    remaining = sum(
        1 for _, r in schedule.iterrows()
        if scraper.race_key(str(r["開催日"]), int(r["レース"])) not in scraper.completed_races
    )
    logging.info("対象: %d レース（%d 開催日）/ 未取得 %d", total_races, schedule["開催日"].nunique(), remaining)

    first_write = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    processed_since_flush = 0
    processed_new = 0

    for idx, row in schedule.iterrows():
        date_str = str(row["開催日"])
        race_no = int(row["レース"])
        if scraper.race_key(date_str, race_no) in scraper.completed_races:
            scraper.stats["skipped"] += 1
            continue

        logging.info(
            "進捗 %d/%d | 開催日 %s | %dR | 今回保存 %d 行",
            processed_new + 1, remaining, date_str, race_no, scraper.stats["rows_written"],
        )

        result = scraper.process_race(date_str, race_no)
        if result is None:
            scraper.finalize_race_batch(first_write=first_write)
            print("\nサイトがアクセス制限中のため中断しました。")
            print("  半日〜24時間空けてから同じコマンドを再実行してください。")
            print_summary(scraper, total_races, target_years)
            sys.exit(2)
        if result is False:
            pass
        else:
            processed_since_flush += 1
            processed_new += 1

        if args.max_races and processed_new >= args.max_races:
            logging.info("--max-races=%d に達したため終了", args.max_races)
            break

        if processed_since_flush >= FLUSH_EVERY_RACES:
            scraper.finalize_race_batch(first_write=first_write)
            first_write = False
            processed_since_flush = 0

    scraper.finalize_race_batch(first_write=first_write)
    print_summary(scraper, total_races, target_years)


if __name__ == "__main__":
    main()
