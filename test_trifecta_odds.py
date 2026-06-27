"""
保存済み3連単オッズ予測モデルを 2025 年データで評価

- モデル: models/3連単オッズ予想/lgbm_trifecta_odds.txt
- データ: 2025 年（レース/選手/気象）
"""
from __future__ import annotations

from train_trifecta_odds import (
    TEST_OUTPUT_DIR,
    TEST_YEARS,
    evaluate_odds,
    load_boat_data,
    load_model,
)

EVALUATION_CSV_PATH = TEST_OUTPUT_DIR / "評価結果.csv"
PREDICTIONS_CSV_PATH = TEST_OUTPUT_DIR / "予測結果.csv"


def main() -> None:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, encoders, features = load_model()
    df = load_boat_data(TEST_YEARS)
    print(f"  行数: {len(df)} / レース数: {len(df) // 6}")

    evaluate_odds(
        model,
        df,
        encoders,
        "2025年テスト",
        feature_names=features,
        output_metrics_path=EVALUATION_CSV_PATH,
        output_predictions_path=PREDICTIONS_CSV_PATH,
    )

    print("\n保存完了:")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
