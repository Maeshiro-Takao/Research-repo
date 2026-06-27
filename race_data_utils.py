"""ComRD.py 用: 年次 CSV の列定義・分割（学習スクリプトは train_race_ranker.py に統合済み）"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_DIR = BASE_DIR / "レースデータ"
PLAYER_DATA_DIR = BASE_DIR / "選手データ"
WEATHER_DATA_DIR = BASE_DIR / "気象データ"

RACE_KEY = ["開催日", "日目", "レース"]
RACE_ROW_KEY = RACE_KEY + ["艇"]
WEATHER_COLS = ["天気", "風向", "風速", "波高"]
RACE_RESULT_COLS = ["着", "登番", "モーター", "ボート", "展示", "3連単オッズ"]

COURSE_STAT_COLS: list[str] = []
for _course in range(1, 7):
    COURSE_STAT_COLS += [
        f"{_course}コース進入回数",
        f"{_course}コース複勝率",
        f"{_course}コース平均スタートタイミング",
        f"{_course}コース平均スタート順位",
    ]
PERIOD_COLS = ["算出期間自", "算出期間至"]

PLAYER_STAT_COLS = (
    ["級", "身長", "体重"]
    + ["勝率", "2連率", "3連率", "当地勝率", "当地2連率", "当地3連率"]
    + ["モーター勝率", "モーター2連率", "モーター3連率"]
    + ["ボート勝率", "ボート2連率", "ボート3連率"]
    + ["平均スタートタイミング"]
    + COURSE_STAT_COLS
    + PERIOD_COLS
)

RACE_OUTPUT_COLS = RACE_ROW_KEY + RACE_RESULT_COLS
PLAYER_OUTPUT_COLS = RACE_ROW_KEY + ["登番", "選手名"] + PLAYER_STAT_COLS
MERGED_OUTPUT_COLS = (
    RACE_ROW_KEY
    + ["登番", "選手名", "着", "モーター", "ボート", "展示", "3連単オッズ"]
    + WEATHER_COLS
    + PLAYER_STAT_COLS
)


def split_merged_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    race_df = df[RACE_OUTPUT_COLS].copy()
    player_df = df[PLAYER_OUTPUT_COLS].copy()
    weather_df = df[RACE_KEY + WEATHER_COLS].drop_duplicates(subset=RACE_KEY).reset_index(drop=True)
    return race_df, player_df, weather_df
