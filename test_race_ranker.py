"""
保存済みレース着順ランキングモデルを 2025 年データで評価

- モデル: models/レース着順ランキング/lgbm_race_ranker.txt
- データ: 2025 年（レース/選手/気象）
- 出力: 3連単 TOP1/5/10/20 的中率、予測結果 CSV、SHAP分析
"""
from __future__ import annotations

from train_race_ranker import (
    RACE_KEY,
    SHAP_SAMPLE_SIZE,
    TEST_OUTPUT_DIR,
    TEST_YEARS,
    evaluate_trifecta,
    load_model,
    load_race_data,
    prepare_dataset,
    save_shap_importance,
    setup_japanese_font,
)
from trifecta_training_utils import run_shap_analysis

EVALUATION_CSV_PATH = TEST_OUTPUT_DIR / "評価結果.csv"
PREDICTIONS_CSV_PATH = TEST_OUTPUT_DIR / "予測結果.csv"
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
    metrics, predictions = evaluate_trifecta(
        model,
        df,
        encoders,
        "2025年テスト",
        feature_names=features,
        output_path=EVALUATION_CSV_PATH,
    )
    predictions.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="UTF-8-sig")

    print("\nSHAP分析（2025年テストデータ）")
    x, _, _, _, _ = prepare_dataset(df, encoders, fit_encoders=False, feature_names=features)
    x_sample = x.sample(min(SHAP_SAMPLE_SIZE, len(x)), random_state=42)
    run_shap_analysis(
        model,
        x_sample,
        SHAP_BEESWARM_PNG,
        SHAP_WATERFALL_PNG,
        title="レース着順ランキング（2025年テスト）",
    )
    save_shap_importance(model, x_sample, SHAP_IMPORTANCE_PNG, SHAP_IMPORTANCE_CSV)

    print("\n保存完了:")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {SHAP_BEESWARM_PNG}")
    print(f"  {SHAP_WATERFALL_PNG}")
    print(f"  {SHAP_IMPORTANCE_PNG}")
    print(f"  {SHAP_IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()
