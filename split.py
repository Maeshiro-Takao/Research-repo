import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "編集データ" / "丸亀学習用_レースデータ.csv"
OUTPUT_DIR = BASE_DIR / "編集データ"

# 7/3 → 0.7、8/2 → 0.8
TRAIN_RATIO = 0.7


def split_timeseries(df: pd.DataFrame, train_ratio: float = 0.7):
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
        "val_ratio": round(1 - train_ratio, 2),
        "total_dates": len(dates),
        "train_dates": len(train_dates),
        "val_dates": len(val_dates),
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


# ============================================================
# メイン
# ============================================================

df = pd.read_csv(DATA_PATH)
print(f"全体: {len(df)} 行 / {len(df) // 6} レース")
print(f"期間: {df['開催日'].min()} 〜 {df['開催日'].max()}")

train_df, val_df, info = split_timeseries(df, train_ratio=TRAIN_RATIO)

ratio_label = "7/3" if TRAIN_RATIO == 0.7 else "8/2"
print(f"\n=== 時系列 {ratio_label} 分割 ===")
print(f"訓練: {info['train_rows']} 行 / {info['train_races']} レース")
print(f"      {info['train_date_from']} 〜 {info['train_date_to']} ({info['train_dates']} 日)")
print(f"検証: {info['val_rows']} 行 / {info['val_races']} レース")
print(f"      {info['val_date_from']} 〜 {info['val_date_to']} ({info['val_dates']} 日)")

# CSV保存
train_path = OUTPUT_DIR / "丸亀学習用_訓練データ.csv"
val_path = OUTPUT_DIR / "丸亀学習用_検証データ.csv"
#info_path = OUTPUT_DIR / "split_info.json"

train_df.to_csv(train_path, index=False, encoding="UTF-8-sig")
val_df.to_csv(val_path, index=False, encoding="UTF-8-sig")
""" with open(info_path, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2) """

print(f"\n保存完了:")
print(f"  {train_path}")
print(f"  {val_path}")
""" print(f"  {info_path}")
 """