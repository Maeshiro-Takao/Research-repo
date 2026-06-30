"""
競艇オッズ保管庫 (orangebuoy.net) から丸亀競艇場の3連単市場オッズを取得する。

対象: 2022〜2025年 / 5分前・10分前オッズ（120通り横持ちCSV）

使い方（1年ずつ・100レースずつ実行）:
    python scrape_market_odds.py --year 2022
    python scrape_market_odds.py --year 2022   # 再実行で続きから100レース
    python scrape_market_odds.py --year 2023
    ...

テスト（1レースのみ）:
    python scrape_market_odds.py --year 2022 --test

1年まとめて取得（非推奨: サイト負荷大）:
    python scrape_market_odds.py --year 2022 --all
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "オッズデータ"
RACE_DATA_DIR = BASE_DIR / "レースデータ"
LOG_FILE = OUTPUT_DIR / "scrape_market_odds.log"
FAILED_LOG = OUTPUT_DIR / "scrape_market_odds_failed.log"

TARGET_YEARS = list(range(2022, 2026))
STADIUM_CODE = 15
STADIUM_NAME = "丸亀"
BASE_URL = "http://www.orangebuoy.net/odds/"

MIN_DELAY_SEC = 3.0
MAX_DELAY_SEC = 8.0
DAY_PAUSE_MIN_SEC = 10.0
DAY_PAUSE_MAX_SEC = 20.0
MAX_RETRIES = 3
RETRY_WAIT_SEC = 15.0
REQUEST_TIMEOUT_SEC = 30
FLUSH_EVERY_RACES = 3
RACES_PER_RUN = 100

TIMING_MODE: dict[str, int] = {
    "5分前": 2,
    "10分前": 3,
}
TIMING_LABELS = list(TIMING_MODE.keys())

RATE_LIMIT_MARKERS = ("アクセスが多いため", "半日ほど時間を空けて")

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}

_shutdown_requested = False


class SiteRateLimitError(Exception):
    """サイト側のアクセス制限"""


def _load_orangebuoy_helpers():
    """既存スクレイパーのHTMLパース処理を再利用する"""
    module_path = BASE_DIR / "オッズ抽出コード" / "ScrapeOrangeBuoyOdds.py"
    spec = importlib.util.spec_from_file_location("orangebuoy_helpers", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"モジュール読み込み失敗: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OB = _load_orangebuoy_helpers()


def combo_column(first: int, second: int, third: int) -> str:
    return f"{first}{second}{third}"


def all_combo_columns() -> list[str]:
    cols: list[str] = []
    for i in range(1, 7):
        for j in range(1, 7):
            if j == i:
                continue
            for k in range(1, 7):
                if k in (i, j):
                    continue
                cols.append(combo_column(i, j, k))
    return cols


COMBO_COLUMNS = all_combo_columns()
BASE_CSV_COLUMNS = ["開催日", "レース番号", "オッズ時点"]
CSV_COLUMNS = BASE_CSV_COLUMNS + COMBO_COLUMNS


def output_path(year: int, timing_label: str) -> Path:
    return OUTPUT_DIR / f"{year}_{STADIUM_NAME}_3連単_{timing_label}.csv"


def progress_path(year: int) -> Path:
    return OUTPUT_DIR / f".scrape_market_odds_{year}.json"


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


def request_shutdown(signum: int, frame) -> None:  # noqa: ARG001
    global _shutdown_requested
    _shutdown_requested = True
    logging.warning("中断信号を受信しました — 保存して終了します...")


def is_rate_limited(text: str) -> bool:
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def build_race_url(date_str: str, race_no: int, mode: int) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return (
        f"{BASE_URL}?day={dt.day}&month={dt.month}&year={dt.year}"
        f"&jyo={STADIUM_CODE}&r={race_no}&mode={mode}"
    )


def odds_map_to_row_values(odds_map: dict[tuple[int, int, int], float]) -> dict[str, float | None]:
    values: dict[str, float | None] = {col: None for col in COMBO_COLUMNS}
    for combo, odds in odds_map.items():
        col = combo_column(*combo)
        values[col] = odds
    return values


def load_race_schedule(year: int) -> pd.DataFrame:
    path = RACE_DATA_DIR / f"丸亀_{year}_レースデータ.csv"
    if not path.exists():
        raise FileNotFoundError(f"レースデータが見つかりません: {path}")
    df = pd.read_csv(path, usecols=["開催日", "レース"])
    df = df.drop_duplicates().sort_values(["開催日", "レース"]).reset_index(drop=True)
    df["開催日"] = df["開催日"].astype(str)
    df["レース"] = df["レース"].astype(int)
    return df


def race_key(date_str: str, race_no: int) -> str:
    return f"{date_str}|{race_no}"


def probe_site_access() -> bool:
    url = build_race_url("2024-01-01", 1, TIMING_MODE["10分前"])
    try:
        resp = requests.get(url, headers=SESSION_HEADERS, timeout=REQUEST_TIMEOUT_SEC)
        resp.encoding = resp.apparent_encoding or "utf-8"
        if is_rate_limited(resp.text):
            logging.error("サイトがアクセス制限中です: %s", resp.text.strip()[:80])
            return False
        if len(resp.text) < 1000:
            logging.warning("応答が短いです（%d bytes）", len(resp.text))
            return False
        return True
    except requests.RequestException as exc:
        logging.error("接続確認失敗: %s", exc)
        return False


class MarketOddsScraper:
    def __init__(self, year: int) -> None:
        self.year = year
        self.session = requests.Session()
        self.session.headers.update(SESSION_HEADERS)
        self.completed_races: set[str] = set()
        self.pending_rows: dict[str, list[dict]] = {label: [] for label in TIMING_LABELS}
        self.stats = {
            "races_ok": 0,
            "races_failed": 0,
            "races_skipped": 0,
            "rows_written": 0,
        }

    def load_progress(self) -> None:
        prog = progress_path(self.year)
        if prog.exists():
            with open(prog, encoding="utf-8") as f:
                data = json.load(f)
            self.completed_races = set(data.get("completed_races", []))
            logging.info("進捗ファイル読込: 取得済み %d レース", len(self.completed_races))

        for label in TIMING_LABELS:
            path = output_path(self.year, label)
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=["開催日", "レース番号"])
                for date_str, race_no in df.drop_duplicates().itertuples(index=False):
                    self.completed_races.add(race_key(str(date_str), int(race_no)))
            except Exception as exc:
                logging.warning("既存CSV読込失敗 (%s): %s", path.name, exc)

        logging.info("取得済みレース（重複除外後）: %d 件", len(self.completed_races))

    def save_progress(self) -> None:
        with open(progress_path(self.year), "w", encoding="utf-8") as f:
            json.dump(
                {"year": self.year, "completed_races": sorted(self.completed_races)},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _sleep_between_requests(self) -> None:
        time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))

    def _sleep_after_day(self) -> None:
        pause = random.uniform(DAY_PAUSE_MIN_SEC, DAY_PAUSE_MAX_SEC)
        logging.info("開催日終了 — %.1f 秒待機", pause)
        time.sleep(pause)

    def fetch_page(self, url: str) -> BeautifulSoup | None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT_SEC)
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

    def fetch_mode_odds(
        self, date_str: str, race_no: int, timing_label: str,
    ) -> dict[tuple[int, int, int], float]:
        mode = TIMING_MODE[timing_label]
        url = build_race_url(date_str, race_no, mode)
        self._sleep_between_requests()
        soup = self.fetch_page(url)
        if soup is None:
            return {}
        if "オッズデータがありません" in soup.get_text():
            return {}

        odds_map = _OB.find_trifecta_odds(soup)
        if len(odds_map) < 100:
            logging.warning(
                "オッズ不足 (%d件): %s %dR %s",
                len(odds_map), date_str, race_no, timing_label,
            )
        return odds_map

    def fetch_race_odds(
        self, date_str: str, race_no: int,
    ) -> dict[str, dict[tuple[int, int, int], float]] | None:
        results: dict[str, dict[tuple[int, int, int], float]] = {}
        for label in TIMING_LABELS:
            try:
                odds_map = self.fetch_mode_odds(date_str, race_no, label)
            except SiteRateLimitError:
                raise
            except Exception as exc:
                logging.error("例外 (%s %dR %s): %s", date_str, race_no, label, exc)
                odds_map = {}
            results[label] = odds_map

        if not any(results[label] for label in TIMING_LABELS):
            return None
        return results

    def build_row(
        self, date_str: str, race_no: int, timing_label: str,
        odds_map: dict[tuple[int, int, int], float],
    ) -> dict:
        row: dict = {
            "開催日": date_str,
            "レース番号": race_no,
            "オッズ時点": timing_label,
        }
        row.update(odds_map_to_row_values(odds_map))
        return row

    def append_rows(self, timing_label: str, rows: list[dict], *, write_header: bool) -> None:
        if not rows:
            return
        path = output_path(self.year, timing_label)
        mode = "w" if write_header else "a"
        with open(path, mode, newline="", encoding="UTF-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        self.stats["rows_written"] += len(rows)

    def flush_pending(self, file_has_header: dict[str, bool]) -> None:
        for label in TIMING_LABELS:
            rows = self.pending_rows[label]
            if not rows:
                continue
            self.append_rows(label, rows, write_header=not file_has_header[label])
            file_has_header[label] = True
            rows.clear()
        self.save_progress()

    def log_failed(self, date_str: str, race_no: int, reason: str) -> None:
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(f"{self.year},{date_str},{race_no},{reason}\n")

    def process_race(self, date_str: str, race_no: int) -> bool | None:
        """
        Returns:
            True  — 成功またはスキップ
            False — データなし
            None  — アクセス制限で中断
        """
        key = race_key(date_str, race_no)
        if key in self.completed_races:
            self.stats["races_skipped"] += 1
            return True

        try:
            odds_by_timing = self.fetch_race_odds(date_str, race_no)
        except SiteRateLimitError as exc:
            logging.error("アクセス制限: %s", str(exc)[:80])
            return None
        except Exception as exc:
            logging.error("レース取得例外 %s %dR: %s", date_str, race_no, exc)
            self.stats["races_failed"] += 1
            self.log_failed(date_str, race_no, str(exc))
            return False

        if odds_by_timing is None:
            self.stats["races_failed"] += 1
            self.log_failed(date_str, race_no, "全時点取得失敗")
            logging.error("取得失敗: %s %dR", date_str, race_no)
            return False

        for label in TIMING_LABELS:
            odds_map = odds_by_timing.get(label, {})
            if not odds_map:
                logging.warning("欠損: %s %dR %s", date_str, race_no, label)
                self.log_failed(date_str, race_no, f"{label}欠損")
            row = self.build_row(date_str, race_no, label, odds_map)
            self.pending_rows[label].append(row)

        self.completed_races.add(key)
        self.stats["races_ok"] += 1
        return True


def file_has_content(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def print_summary(
    scraper: MarketOddsScraper,
    batch_races: int,
    remaining_races: int,
) -> None:
    s = scraper.stats
    print(f"\n=== 取得サマリ ({scraper.year}年) ===")
    print(f"  今回の対象: {batch_races} レース")
    print(f"  取得成功: {s['races_ok']}")
    print(f"  取得失敗: {s['races_failed']}")
    print(f"  スキップ（取得済み）: {s['races_skipped']}")
    print(f"  保存行数: {s['rows_written']}")
    print(f"  残り未取得: {remaining_races} レース")
    for label in TIMING_LABELS:
        print(f"  出力: {output_path(scraper.year, label)}")
    if remaining_races > 0:
        print(f"\n  続きを取得する場合（{RACES_PER_RUN}レースずつ）:")
        print(f"    python scrape_market_odds.py --year {scraper.year}")


def run_scrape(
    year: int,
    *,
    test_mode: bool = False,
    max_races: int | None = None,
    target_date: str | None = None,
) -> int:
    schedule = load_race_schedule(year)
    if target_date:
        schedule = schedule.loc[schedule["開催日"] == target_date]
    if test_mode:
        first = schedule.iloc[0]
        schedule = schedule.loc[
            (schedule["開催日"] == first["開催日"])
            & (schedule["レース"] == first["レース"])
        ]

    scraper = MarketOddsScraper(year)
    scraper.load_progress()

    keys = schedule["開催日"].astype(str) + "|" + schedule["レース"].astype(str)
    all_pending = schedule.loc[~keys.isin(scraper.completed_races)].copy()
    pending = all_pending
    if not test_mode and max_races is not None:
        pending = pending.head(max_races)

    batch_races = len(pending)
    logging.info(
        "%d年 開始 — 今回 %d レース / 残り %d / 全 %d レース",
        year, batch_races, len(all_pending), len(schedule),
    )

    header_state = {label: file_has_content(output_path(year, label)) for label in TIMING_LABELS}
    processed = 0
    prev_date: str | None = None
    races_on_day = 0

    for _, row in pending.iterrows():
        if _shutdown_requested:
            break

        date_str = str(row["開催日"])
        race_no = int(row["レース"])
        processed += 1

        logging.info(
            "進捗 [%d年] %d/%d | 開催日 %s | %dR",
            year, processed, batch_races, date_str, race_no,
        )

        if prev_date is not None and date_str != prev_date and races_on_day > 0:
            scraper.flush_pending(header_state)
            scraper._sleep_after_day()
            races_on_day = 0

        result = scraper.process_race(date_str, race_no)
        if result is None:
            scraper.flush_pending(header_state)
            print("\nサイトがアクセス制限中のため中断しました。")
            print("  半日〜24時間空けてから同じコマンドを再実行してください。")
            remaining_races = max(len(all_pending) - scraper.stats["races_ok"], 0)
            print_summary(scraper, batch_races, remaining_races)
            return 2

        if result:
            races_on_day += 1
            if processed % FLUSH_EVERY_RACES == 0:
                scraper.flush_pending(header_state)

        prev_date = date_str

    if prev_date is not None and races_on_day > 0:
        scraper.flush_pending(header_state)

    remaining_races = max(len(all_pending) - scraper.stats["races_ok"], 0)
    print_summary(scraper, batch_races, remaining_races)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="丸亀3連単市場オッズスクレイパー（2022〜2025）")
    parser.add_argument("--year", type=int, required=True, choices=TARGET_YEARS, help="取得する年")
    parser.add_argument("--test", action="store_true", help="1レースのみ試行")
    parser.add_argument("--date", type=str, help="特定日のみ (YYYY-MM-DD)")
    parser.add_argument(
        "--max-races",
        type=int,
        default=RACES_PER_RUN,
        help=f"今回取得する最大レース数（デフォルト: {RACES_PER_RUN}）",
    )
    parser.add_argument("--all", action="store_true", help="レース数制限なしで1年分を取得（非推奨）")
    parser.add_argument("--probe-only", action="store_true", help="接続確認のみ")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    setup_logging()
    max_races = None if args.all else args.max_races
    logging.info(
        "設定: 1回 %s / リクエスト間隔 %.0f〜%.0f秒 / 開催日後 %.0f〜%.0f秒 / 最大リトライ %d回",
        f"{max_races}レース" if max_races else "制限なし",
        MIN_DELAY_SEC, MAX_DELAY_SEC, DAY_PAUSE_MIN_SEC, DAY_PAUSE_MAX_SEC, MAX_RETRIES,
    )

    if not probe_site_access():
        print("\nサイトがアクセス制限中です。半日〜24時間空けてから再実行してください。")
        sys.exit(2)

    if args.probe_only:
        print("接続OK — スクレイピング可能です。")
        return

    exit_code = run_scrape(
        args.year,
        test_mode=args.test,
        max_races=max_races,
        target_date=args.date,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
