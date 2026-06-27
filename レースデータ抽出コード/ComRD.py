"""
丸亀の年次レースデータを統合して出力する。

各モジュールの役割:
  ExtRD.py : txt からの抽出のみ
  CalcRD.py: 特徴量計算のみ
  ComRD.py : 統合と CSV 出力のみ

出力:
  レースデータ/丸亀_{年}_レースデータ.csv
  選手データ/丸亀_{年}_選手データ.csv
  気象データ/丸亀_{年}_気象データ.csv

使い方:
    python レースデータ抽出コード/ExtRD.py   # グレードCSV生成（初回・更新時）
    python レースデータ抽出コード/ComRD.py   # レースデータ生成
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from CalcRD import compute  # noqa: E402
from ExtRD import RACE_ROW_KEY, extract, load_grade_race_table  # noqa: E402
from race_data_utils import (  # noqa: E402
    MERGED_OUTPUT_COLS,
    PLAYER_DATA_DIR,
    RACE_DATA_DIR,
    RACE_KEY,
    WEATHER_DATA_DIR,
    split_merged_dataframe,
)

OUTPUT_COLUMNS = MERGED_OUTPUT_COLS

TRAIN_YEARS = range(14, 25)
TEST_YEARS = range(25, 26)
LEGACY_COMBINED_GLOB = "丸亀_*_レースデータ.csv"


def player_years_for(years: range) -> range:
    """レース年に対応する fan*.txt の読み込み範囲（翌年フォルダを含む）"""
    return range(years.start, years.stop + 1)


def merge_extract_and_computed(
    extracted: pd.DataFrame,
    computed: pd.DataFrame,
) -> pd.DataFrame:
    """抽出DataFrameと計算DataFrameを結合（重複レコードなし）"""
    if len(extracted) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    merged = extracted.merge(
        computed,
        on=RACE_ROW_KEY,
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(extracted):
        raise ValueError(
            f"結合後の行数が一致しません: 抽出={len(extracted)}, 結合後={len(merged)}"
        )

    missing = [col for col in OUTPUT_COLUMNS if col not in merged.columns]
    if missing:
        raise KeyError(f"出力列が不足しています: {missing}")

    return merged[OUTPUT_COLUMNS]


def history_years_for(years: range) -> range:
    """勝率計算用の全国成績履歴読込年（当地用に12年〜）"""
    return range(12, years.stop)


def build_dataset(
    years: range,
    grade_df: pd.DataFrame,
    rate_history_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("  txt からデータを抽出 (ExtRD)...")
    extracted, player_records = extract(
        years,
        player_years=player_years_for(years),
    )

    print("  特徴量を計算 (CalcRD)...")
    computed, rate_meta = compute(
        extracted,
        player_records,
        grade_df,
        rate_history_df=rate_history_df,
        history_years=history_years_for(years),
    )

    print("  抽出データと計算データを結合...")
    merged = merge_extract_and_computed(extracted, computed)
    return merged, rate_meta


def save_yearly_files(df: pd.DataFrame, label: str) -> None:
    if len(df) == 0:
        print(f"  {label}: データなし")
        return

    RACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLAYER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    years = pd.to_datetime(df["開催日"]).dt.year
    for year in sorted(years.unique()):
        year_df = df.loc[years == year].copy()
        race_df, player_df, weather_df = split_merged_dataframe(year_df)

        race_path = RACE_DATA_DIR / f"丸亀_{int(year)}_レースデータ.csv"
        player_path = PLAYER_DATA_DIR / f"丸亀_{int(year)}_選手データ.csv"
        weather_path = WEATHER_DATA_DIR / f"丸亀_{int(year)}_気象データ.csv"

        race_df.to_csv(race_path, index=False, encoding="UTF-8-sig")
        player_df.to_csv(player_path, index=False, encoding="UTF-8-sig")
        weather_df.to_csv(weather_path, index=False, encoding="UTF-8-sig")

        race_count = year_df.drop_duplicates(RACE_KEY).shape[0]
        print(f"\n  --- {int(year)}年 ---")
        print(f"  行数: {len(year_df)} / レース数: {race_count}")
        print(f"  レース: {race_path.name} ({len(race_df.columns)}列)")
        print(f"  選手: {player_path.name} ({len(player_df.columns)}列)")
        print(f"  気象: {weather_path.name} ({len(weather_df.columns)}列, {len(weather_df)}行)")
        print(
            "  3連単オッズあり:",
            int(year_df["3連単オッズ"].notna().sum()),
            f"({year_df['3連単オッズ'].notna().mean():.1%})",
        )
        print(
            "  当地勝率あり:",
            int(year_df["当地勝率"].notna().sum()),
            f"({year_df['当地勝率'].notna().mean():.1%})",
        )


def remove_legacy_outputs() -> None:
    legacy_names = ("丸亀学習用_レースデータ.csv", "丸亀テスト用_レースデータ.csv")
    for name in legacy_names:
        path = RACE_DATA_DIR / name
        if path.exists():
            path.unlink()
            print(f"  旧ファイル削除: {path.name}")


def main():
    grade_df = load_grade_race_table()
    print(f"  グレードCSV: {len(grade_df)} レース")
    print(f"  統合列数: {len(OUTPUT_COLUMNS)}")
    print(f"  出力先: レースデータ/ 選手データ/ 気象データ/")

    rate_history_df: pd.DataFrame | None = None

    print(f"\n=== 学習期間 ({TRAIN_YEARS.start + 2000}〜{TRAIN_YEARS.stop - 1 + 2000}) ===")
    train_df, rate_meta = build_dataset(TRAIN_YEARS, grade_df, rate_history_df)
    save_yearly_files(train_df, "学習期間")
    rate_history_df = rate_meta.copy()

    print(f"\n=== テスト期間 ({TEST_YEARS.start + 2000}) ===")
    test_df, _ = build_dataset(TEST_YEARS, grade_df, rate_history_df)
    save_yearly_files(test_df, "テスト期間")

    remove_legacy_outputs()

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    print("\n=== 全体 ===")
    print("総件数:", len(all_df))
    print("開催日数:", all_df["開催日"].nunique())
    print("レース数:", all_df.drop_duplicates(RACE_KEY).shape[0])
    dup = all_df.duplicated(RACE_ROW_KEY + ["登番"]).sum()
    print("重複行:", dup)


if __name__ == "__main__":
    main()
