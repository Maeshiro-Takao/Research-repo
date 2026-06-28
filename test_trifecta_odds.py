"""
保存済み3連単オッズ予測モデルを 2025 年データで評価

- モデル: models/3連単オッズ予想/combinations/
- データ: 2025年レース/選手/気象 + オッズデータ/丸亀テスト用_3連単オッズ.csv
"""
from __future__ import annotations

from pathlib import Path

from train_trifecta_odds import (
    TEST_OUTPUT_DIR,
    TEST_YEARS,
    build_labeled_frame,
    evaluate_and_save,
    load_models,
    load_odds_csv,
    load_race_frame,
    predict_labeled_races,
)
from trifecta_training_utils import setup_japanese_font

EVALUATION_CSV_PATH = TEST_OUTPUT_DIR / "評価結果.csv"
PREDICTIONS_CSV_PATH = TEST_OUTPUT_DIR / "予測結果.csv"
TEST_ODDS_PATH = Path(__file__).resolve().parent / "オッズデータ" / "丸亀テスト用_3連単オッズ.csv"


def main() -> None:
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("モデル読み込み...")
    models, encoders, _ = load_models()
    print(f"  読み込みモデル数: {len(models)}")

    print("\nテストデータ読み込み: 2025年")
    race_df = load_race_frame(TEST_YEARS)
    odds_df = load_odds_csv(TEST_ODDS_PATH)
    print(f"  行数: {len(race_df)} / レース数: {len(race_df) // 6}")
    print(f"  オッズ行数: {len(odds_df)}")

    test_frame, _ = build_labeled_frame(race_df, odds_df, encoders, fit_encoders=False)
    pred_df = predict_labeled_races(models, test_frame)
    evaluate_and_save(
        pred_df,
        "2025年テスト",
        EVALUATION_CSV_PATH,
        PREDICTIONS_CSV_PATH,
    )

    print("\n保存完了:")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
