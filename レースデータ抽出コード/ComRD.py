"""
丸亀の学習用/テスト用レースデータを統合して出力する。

各モジュールの役割:
  ExtRD: データ抽出（競走成績・選手データ・グレードCSV）
  CalcRD: 計算（直近勝率・連帯率・当地勝率）
  ComRD: 統合出力（本スクリプト）

使い方:
    python レースデータ抽出コード/ExtRD.py   # グレードCSV生成（初回・更新時）
    python レースデータ抽出コード/ComRD.py   # レースデータ生成
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ExtRD import (  # noqa: E402
    extract_player_records,
    extract_race_dataframe,
    load_grade_race_table,
)
from CalcRD import (  # noqa: E402
    add_local_win_rate,
    attach_player_stats,
)

OUTPUT_DIR = BASE_DIR / "レースデータ"

OUTPUT_COLUMNS = [
    "開催日", "日目", "レース",
    "着", "艇",
    "登番", "選手名", "モーター", "ボート", "展示",
    "天気", "風向", "風速", "波高",
    "3連単オッズ", "当地勝率",
    "級", "身長", "体重", "勝率", "複勝率",
    "直近10年当地勝率", "直近10年2連帯率", "直近10年3連帯率",
    "直近5年当地勝率", "直近5年2連帯率", "直近5年3連帯率",
    "1着率", "2着率", "3着率",
    "優出回数", "優勝回数", "平均スタートタイミング",
]
for _course in range(1, 7):
    OUTPUT_COLUMNS += [
        f"{_course}コース進入回数",
        f"{_course}コース複勝率",
        f"{_course}コース平均スタートタイミング",
        f"{_course}コース平均スタート順位",
    ]
OUTPUT_COLUMNS += ["算出期間自", "算出期間至"]

DATASETS = [
    {"years": range(14, 25), "output": "丸亀学習用_レースデータ.csv"},
    {"years": range(25, 26), "output": "丸亀テスト用_レースデータ.csv"},
]


def player_years_for(years: range) -> range:
    """レース年に対応する fan*.txt の読み込み範囲（翌年フォルダを含む）"""
    return range(years.start, years.stop + 1)


def build_dataset(
    years: range,
    grade_df: pd.DataFrame,
    player_history: list[dict] | None,
    rate_history_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("  競走成績からレース・気象・オッズを抽出...")
    race_df = extract_race_dataframe(years)

    print("  当地勝率を算出（グレードCSV使用）...")
    race_df, rate_meta = add_local_win_rate(race_df, grade_df, rate_history_df)

    print("  選手データを付与...")
    merged = attach_player_stats(
        race_df,
        years,
        player_history,
        player_years=player_years_for(years),
    )

    output_cols = OUTPUT_COLUMNS
    for col in output_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged[output_cols], rate_meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grade_df = load_grade_race_table()
    print(f"  グレードCSV: {len(grade_df)} レース")

    player_history: list[dict] | None = None
    rate_history_df: pd.DataFrame | None = None

    for dataset in DATASETS:
        print(f"\n=== {dataset['output']} ===")
        df, rate_meta = build_dataset(
            dataset["years"],
            grade_df,
            player_history,
            rate_history_df,
        )

        output_path = OUTPUT_DIR / dataset["output"]
        df.to_csv(output_path, index=False, encoding="UTF-8-sig")

        print("総件数:", len(df))
        if len(df) > 0:
            print("開催日数:", df["開催日"].nunique())
            race_count = df.drop_duplicates(["開催日", "日目", "レース"]).shape[0]
            print("レース数:", race_count)
            if "grade" in rate_meta.columns:
                print("グレード内訳:", rate_meta["grade"].value_counts().to_dict())
            print(
                "3連単オッズあり:",
                int(df["3連単オッズ"].notna().sum()),
                f"({df['3連単オッズ'].notna().mean():.1%})",
            )
            print(
                "当地勝率あり:",
                int(df["当地勝率"].notna().sum()),
                f"({df['当地勝率'].notna().mean():.1%})",
            )
            print(
                "選手統計(勝率)あり:",
                int(df["勝率"].notna().sum()),
                f"({df['勝率'].notna().mean():.1%})",
            )
            print("保存:", output_path)

        if dataset["output"] == "丸亀学習用_レースデータ.csv":
            player_history = extract_player_records(player_years_for(dataset["years"]))
            rate_history_df = rate_meta.copy()


if __name__ == "__main__":
    main()
