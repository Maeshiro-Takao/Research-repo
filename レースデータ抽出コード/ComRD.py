"""
丸亀の学習用/テスト用レースデータを統合して出力する。

各モジュールの役割:
  ExtRD.py : txt からの抽出のみ
  CalcRD.py: 特徴量計算のみ
  ComRD.py : 統合と CSV 出力のみ

処理フロー:
  元データ/*.txt → ExtRD.py → 抽出DataFrame
  抽出DataFrame  → CalcRD.py → 計算DataFrame
  抽出 + 計算    → ComRD.py  → 学習/テストCSV

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

from CalcRD import COMPUTED_COLUMNS, compute  # noqa: E402
from ExtRD import EXTRACT_COLUMNS, RACE_ROW_KEY, extract, load_grade_race_table  # noqa: E402

OUTPUT_DIR = BASE_DIR / "レースデータ"

OUTPUT_COLUMNS = EXTRACT_COLUMNS + COMPUTED_COLUMNS

DATASETS = [
    {"years": range(14, 25), "output": "丸亀学習用_レースデータ.csv"},
    {"years": range(25, 26), "output": "丸亀テスト用_レースデータ.csv"},
]


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
    return merged[OUTPUT_COLUMNS]


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
    )

    print("  抽出データと計算データを結合...")
    merged = merge_extract_and_computed(extracted, computed)
    return merged, rate_meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grade_df = load_grade_race_table()
    print(f"  グレードCSV: {len(grade_df)} レース")

    rate_history_df: pd.DataFrame | None = None

    for dataset in DATASETS:
        print(f"\n=== {dataset['output']} ===")
        df, rate_meta = build_dataset(
            dataset["years"],
            grade_df,
            rate_history_df,
        )

        output_path = OUTPUT_DIR / dataset["output"]
        df.to_csv(output_path, index=False, encoding="UTF-8-sig")

        print("総件数:", len(df))
        if len(df) > 0:
            print("開催日数:", df["開催日"].nunique())
            race_count = df.drop_duplicates(RACE_ROW_KEY[:3]).shape[0]
            print("レース数:", race_count)
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
            dup = df.duplicated(RACE_ROW_KEY + ["登番"]).sum()
            print("重複行:", dup)
            print("保存:", output_path)

        if dataset["output"] == "丸亀学習用_レースデータ.csv":
            rate_history_df = rate_meta.copy()


if __name__ == "__main__":
    main()
