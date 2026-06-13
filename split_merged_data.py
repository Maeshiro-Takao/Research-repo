"""
丸亀学習用_レースデータ.csv と 丸亀学習用_選手データ.csv を結合し、
開催日の時系列順で訓練用/検証用に分割して CSV 保存する。

使い方:
    python split_merged_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "編集データ" / "丸亀学習用_レースデータ.csv"
PLAYER_DATA_PATH = BASE_DIR / "編集データ" / "丸亀学習用_選手データ.csv"
OUTPUT_DIR = BASE_DIR / "編集データ"

TRAIN_PATH = OUTPUT_DIR / "丸亀学習用_訓練データ.csv"
VALID_PATH = OUTPUT_DIR / "丸亀学習用_検証データ.csv"

RACE_ROW_KEY = ["開催日", "日目", "レース", "艇"]
PLAYER_META_COLS = ["名前漢字", "算出期間自", "算出期間至"]
TRAIN_RATIO = 0.7


def merge_player_data(race_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """登番と開催日（算出期間）で選手データを結合する"""
    race = race_df.copy()
    player = player_df.copy()
    race["開催日"] = pd.to_datetime(race["開催日"])

    if {"算出期間自", "算出期間至"}.issubset(player.columns):
        player["算出期間自"] = pd.to_datetime(player["算出期間自"])
        player["算出期間至"] = pd.to_datetime(player["算出期間至"])
        player_cols = [c for c in player.columns if c != "登番"]
        merged = race.merge(player, on="登番", how="left")
        period_match = (
            merged["算出期間自"].notna()
            & (merged["開催日"] >= merged["算出期間自"])
            & (merged["開催日"] <= merged["算出期間至"])
        )
        matched = (
            merged.loc[period_match, RACE_ROW_KEY + player_cols]
            .drop_duplicates(RACE_ROW_KEY)
        )
        out = race.merge(matched, on=RACE_ROW_KEY, how="left")
    else:
        player = player.drop_duplicates(subset=["登番"], keep="last")
        player_cols = [c for c in player.columns if c != "登番"]
        out = race.merge(player, on="登番", how="left")

    return out.drop(columns=[c for c in PLAYER_META_COLS if c in out.columns])


def split_timeseries(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """開催日の時系列順で学習用/検証用に分割する"""
    df = df.copy()
    df["開催日"] = pd.to_datetime(df["開催日"])

    dates = np.sort(df["開催日"].unique())
    split_idx = int(len(dates) * train_ratio)
    train_dates = set(dates[:split_idx])
    val_dates = set(dates[split_idx:])

    train_df = df[df["開催日"].isin(train_dates)].copy()
    val_df = df[df["開催日"].isin(val_dates)].copy()

    info = {
        "train_ratio": train_ratio,
        "train_date_from": str(train_df["開催日"].min().date()),
        "train_date_to": str(train_df["開催日"].max().date()),
        "val_date_from": str(val_df["開催日"].min().date()),
        "val_date_to": str(val_df["開催日"].max().date()),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_races": len(train_df) // 6,
        "val_races": len(val_df) // 6,
    }
    return train_df, val_df, info


def main():
    print(f"データ読み込み: {RACE_DATA_PATH.name}, {PLAYER_DATA_PATH.name}")
    race_df = pd.read_csv(RACE_DATA_PATH)
    player_df = pd.read_csv(PLAYER_DATA_PATH)
    df = merge_player_data(race_df, player_df)

    matched = df["級"].notna().sum() if "級" in df.columns else 0
    print(f"  全体: {len(df)} 行 / {len(df) // 6} レース")
    print(f"  選手データ結合: {matched} 行 ({matched / len(df):.1%})")
    print(f"  期間: {df['開催日'].min()} 〜 {df['開催日'].max()}")

    train_df, val_df, info = split_timeseries(df)
    ratio_label = "7/3" if TRAIN_RATIO == 0.7 else f"{int(TRAIN_RATIO * 10)}/{int((1 - TRAIN_RATIO) * 10)}"
    print(f"\n=== 時系列 {ratio_label} 分割 ===")
    print(f"訓練: {info['train_rows']} 行 / {info['train_races']} レース")
    print(f"      {info['train_date_from']} 〜 {info['train_date_to']}")
    print(f"検証: {info['val_rows']} 行 / {info['val_races']} レース")
    print(f"      {info['val_date_from']} 〜 {info['val_date_to']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False, encoding="UTF-8-sig")
    val_df.to_csv(VALID_PATH, index=False, encoding="UTF-8-sig")

    print("\n保存完了:")
    print(f"  {TRAIN_PATH}")
    print(f"  {VALID_PATH}")


if __name__ == "__main__":
    main()
