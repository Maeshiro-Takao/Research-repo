"""
二段階3連単オッズモデル — 2025年テスト評価
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from train_trifecta_odds import TEST_YEARS, load_boat_data
from train_trifecta_odds_two_stage import (
    EVAL_METRICS,
    TEST_OUTPUT_DIR,
    evaluate_two_stage,
    load_two_stage_model,
    save_confusion_matrix,
)

EVALUATION_CSV_PATH = TEST_OUTPUT_DIR / "評価結果.csv"
PREDICTIONS_CSV_PATH = TEST_OUTPUT_DIR / "予測結果.csv"
CONFUSION_MATRIX_CSV_PATH = TEST_OUTPUT_DIR / "混同行列.csv"
CONFUSION_MATRIX_PNG = TEST_OUTPUT_DIR / "混同行列.png"


def main() -> None:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classifier, regressors, global_regressor, encoders, features = load_two_stage_model()
    df = load_boat_data(TEST_YEARS)
    print(f"テストデータ: 行数 {len(df)} / レース数 {len(df) // 6}")

    ts_reg, ts_clf, pred_df = evaluate_two_stage(
        classifier, regressors, global_regressor, df, encoders,
        "2025年テスト", feature_names=features,
    )
    save_confusion_matrix(
        np.array(ts_clf["confusion_matrix"]),
        CONFUSION_MATRIX_PNG,
        CONFUSION_MATRIX_CSV_PATH,
        "混同行列（2025年テスト）",
    )
    pd.DataFrame([{**ts_reg, "accuracy": ts_clf["accuracy"], "macro_f1": ts_clf["macro_f1"]}]).to_csv(
        EVALUATION_CSV_PATH, index=False, encoding="UTF-8-sig",
    )
    pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="UTF-8-sig")

    print("\n=== 2025年テスト ===")
    for key in EVAL_METRICS:
        val = ts_reg[key]
        if "rate" in key:
            print(f"  {key}: {val:.2%}")
        elif key in ("rmse_odds", "mae_odds"):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val:.4f}")
    print(f"  accuracy: {ts_clf['accuracy']:.4f}")
    print(f"  macro_f1: {ts_clf['macro_f1']:.4f}")

    print("\n保存完了:")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
