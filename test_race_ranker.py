"""
保存済み 6艇ランキング + Plackett–Luce 3連単予想モデルを 2025 年データで評価

- モデル: models/3連単予想/lgbm_race_ranker.txt
- データ: 2025 年（レース/選手/気象）
- 出力: NDCG@1/3, 3連単 TOP1/5/10/20, 予測結果 CSV, SHAP
"""
from __future__ import annotations

import pandas as pd

from train_race_ranker import (
    RACE_KEY,
    SHAP_SAMPLE_SIZE,
    TEST_OUTPUT_DIR,
    TEST_YEARS,
    evaluate_model,
    load_model,
    load_race_data,
    prepare_dataset,
    save_shap_importance,
)
from trifecta_training_utils import run_shap_analysis, setup_japanese_font

SHAP_BEESWARM_PNG = TEST_OUTPUT_DIR / "特徴量影響方向.png"
SHAP_WATERFALL_PNG = TEST_OUTPUT_DIR / "特徴量ウォーターフォール.png"
SHAP_IMPORTANCE_PNG = TEST_OUTPUT_DIR / "特徴量重要度.png"
SHAP_IMPORTANCE_CSV = TEST_OUTPUT_DIR / "特徴量重要度.csv"


def load_test_data() -> pd.DataFrame:
    years = list(TEST_YEARS)
    print(f"テストデータ読み込み: {years}年（レース/選手/気象）")
    df = load_race_data(years)
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    df = df.loc[sizes == 6].copy()
    print(f"  行数: {len(df)} / レース数: {len(df) // 6}")
    return df


def main() -> None:
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, encoders, features = load_model()
    df = load_test_data()
    evaluate_model(
        model,
        df,
        encoders,
        "2025年テスト",
        feature_names=features,
        output_dir=TEST_OUTPUT_DIR,
    )

    print("\nSHAP分析（2025年テストデータ）")
    x, _, _, _, _ = prepare_dataset(df, encoders, fit_encoders=False, feature_names=features)
    x_sample = x.sample(min(SHAP_SAMPLE_SIZE, len(x)), random_state=42)
    run_shap_analysis(
        model,
        x_sample,
        SHAP_BEESWARM_PNG,
        SHAP_WATERFALL_PNG,
        title="6艇ランキング_3連単予想（2025年テスト）",
    )
    save_shap_importance(model, x_sample, SHAP_IMPORTANCE_PNG, SHAP_IMPORTANCE_CSV)

    print("\n保存完了:")
    print(f"  {TEST_OUTPUT_DIR / '評価結果.csv'}")
    print(f"  {TEST_OUTPUT_DIR / '予測結果.csv'}")
    print(f"  {TEST_OUTPUT_DIR / '予測結果_120通り.csv'}")
    print(f"  {SHAP_BEESWARM_PNG}")
    print(f"  {SHAP_WATERFALL_PNG}")
    print(f"  {SHAP_IMPORTANCE_PNG}")
    print(f"  {SHAP_IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()
