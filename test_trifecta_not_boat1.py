"""
1着≠1号艇のレースを対象に、学習済みモデルで3連単を予測・評価する。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from trifecta_training_utils import (
    CATEGORICAL_FEATURES,
    build_feature_list,
    encode_categorical_columns,
    run_shap_analysis,
    setup_japanese_font,
)

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "レースデータ" / "丸亀テスト用_レースデータ.csv"
MODEL_DIR = BASE_DIR / "models" / "1着1号艇以外予想"
TEST_OUTPUT_DIR = MODEL_DIR / "テスト"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgbm_trifecta_not_boat1_model.txt"
ENCODER_PATH = MODEL_DIR / "trifecta_not_boat1_encoders.pkl"
PREDICTIONS_CSV_PATH = TEST_OUTPUT_DIR / "予測結果.csv"
EVALUATION_CSV_PATH = TEST_OUTPUT_DIR / "評価結果.csv"
SHAP_BEESWARM_PNG_PATH = TEST_OUTPUT_DIR / "特徴量影響方向.png"
SHAP_WATERFALL_PNG_PATH = TEST_OUTPUT_DIR / "特徴量ウォーターフォール.png"

RACE_KEY = ["開催日", "日目", "レース"]
TOP_N_LIST = [1, 3, 5, 10, 30]
SHAP_SAMPLE_SIZE = 2000

FEATURES: list[str] = []


def configure_features(df: pd.DataFrame) -> None:
    global FEATURES
    FEATURES = build_feature_list(df)


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    return df.loc[sizes == 6].copy()


def load_test_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}")
    df = pd.read_csv(RACE_DATA_PATH, low_memory=False)
    configure_features(df)
    df = filter_complete_races(df)
    matched = df["級"].notna().sum() if "級" in df.columns else 0
    print(f"  全体: {len(df)} 行 / {len(df) // 6} レース")
    print(f"  選手データ結合: {matched} 行 ({matched / len(df):.1%})")
    print(f"  特徴量数: {len(FEATURES)}")
    print(
        f"  期間: {pd.to_datetime(df['開催日']).min().date()} 〜 "
        f"{pd.to_datetime(df['開催日']).max().date()}"
    )
    return df


def load_model_and_encoder() -> tuple[lgb.Booster, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {ENCODER_PATH}")
    return lgb.Booster(model_file=str(MODEL_PATH)), joblib.load(ENCODER_PATH)


def prepare_all(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = filter_complete_races(df)
    df = df.sort_values(RACE_KEY).reset_index(drop=True)
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    df = df.drop(columns=["日目", "選手名"], errors="ignore")
    for col in CATEGORICAL_FEATURES:
        if col not in encoders:
            raise KeyError(
                f"エンコーダに {col} がありません。"
                "先に train_trifecta_not_boat1.py を実行してください。"
            )
    df, encoders = encode_categorical_columns(df, encoders, update_encoders=False)
    x = df[FEATURES].copy()
    x.columns = FEATURES
    return x


def prepare_race(
    race_df: pd.DataFrame,
    features: list[str],
    encoders: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = race_df.copy()
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    boat_numbers = df["艇"].astype(int).values
    for col in CATEGORICAL_FEATURES:
        if col not in encoders:
            raise KeyError(f"エンコーダに {col} がありません")
    df, encoders = encode_categorical_columns(df, encoders, update_encoders=False)
    for col in features:
        if col not in df.columns:
            df[col] = pd.NA
    return df[features], boat_numbers


def calc_trifecta_probs_from_scores(scores: np.ndarray) -> dict[tuple[int, int, int], float]:
    n = len(scores)
    exp_scores = np.exp(scores - scores.max())
    results: dict[tuple[int, int, int], float] = {}
    for i in range(n):
        p1 = max(exp_scores[i] / exp_scores.sum(), 1e-12)
        rem1 = [b for b in range(n) if b != i]
        sum_rem1 = exp_scores[rem1].sum()
        for j in rem1:
            p2 = max(exp_scores[j] / sum_rem1, 1e-12)
            rem2 = [b for b in rem1 if b != j]
            sum_rem2 = exp_scores[rem2].sum()
            for k in rem2:
                p3 = max(exp_scores[k] / sum_rem2, 1e-12)
                results[(i + 1, j + 1, k + 1)] = p1 * p2 * p3
    return results


def get_actual_trifecta(race_df: pd.DataFrame) -> tuple[int, int, int]:
    top3 = race_df.sort_values("着").head(3)
    return tuple(top3["艇"].astype(int).tolist())  # type: ignore[return-value]


def predict_race_trifecta(
    model: lgb.Booster,
    race_df: pd.DataFrame,
    encoders: dict,
) -> tuple[tuple[int, int, int], list[tuple[tuple[int, int, int], float]]]:
    features = model.feature_name()
    x, boat_numbers = prepare_race(race_df, features, encoders)
    scores = model.predict(x)
    score_arr = np.zeros(6)
    for idx, boat in enumerate(boat_numbers):
        score_arr[boat - 1] = scores[idx]
    ranking = sorted(
        calc_trifecta_probs_from_scores(score_arr).items(),
        key=lambda x: x[1], reverse=True,
    )
    return ranking[0][0], ranking


def evaluate_trifecta(model, df, encoders, label, output_path: Path, predictions_path: Path):
    hits = {n: 0 for n in TOP_N_LIST}
    first_hits = second_hits = 0
    logloss_sum = 0.0
    n_races = 0
    rows = []

    for key, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        actual = get_actual_trifecta(race_df)
        pred_top, ranking = predict_race_trifecta(model, race_df, encoders)
        prob_map = dict(ranking)
        for n in TOP_N_LIST:
            if actual in [c for c, _ in ranking[:n]]:
                hits[n] += 1
        first_hits += pred_top[0] == actual[0]
        second_hits += pred_top[:2] == actual[:2]
        logloss_sum += -np.log(max(prob_map.get(actual, 1e-12), 1e-12))
        n_races += 1
        rows.append({
            "開催日": key[0],
            "レース": key[2],
            "実際3連単": "-".join(map(str, actual)),
            "予測3連単": "-".join(map(str, pred_top)),
            "予測確率": ranking[0][1],
            "的中": pred_top == actual,
            "TOP10": " / ".join(
                f"{'-'.join(map(str, c))}({p:.4f})" for c, p in ranking[:10]
            ),
        })

    print(f"\n=== 3連単評価 ({label}) ===")
    print(f"  レース数: {n_races}")
    for n in TOP_N_LIST:
        print(f"  3連単 TOP{n} 的中率: {hits[n] / n_races:.2%}")
    print(f"  1着艇番 的中率: {first_hits / n_races:.2%}")
    print(f"  2連単 的中率:   {second_hits / n_races:.2%}")
    print(f"  3連単 的中率:   {hits[1] / n_races:.2%}")
    print(f"  平均-log確率: {logloss_sum / n_races:.4f}")

    metrics = {
        "label": label,
        "n_races": n_races,
        "first_hit_rate": first_hits / n_races,
        "exacta_hit_rate": second_hits / n_races,
        "trifecta_top1_rate": hits[1] / n_races,
        "trifecta_avg_neg_log_prob": logloss_sum / n_races,
        **{f"trifecta_top{n}_rate": hits[n] / n_races for n in TOP_N_LIST},
    }
    pd.DataFrame([metrics]).to_csv(output_path, index=False, encoding="UTF-8-sig")
    pd.DataFrame(rows).to_csv(predictions_path, index=False, encoding="UTF-8-sig")
    print(f"  評価結果CSV: {output_path}")
    print(f"  予測結果CSV: {predictions_path}")
    return metrics


def main():
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    model, encoders = load_model_and_encoder()
    print(f"  モデル: {MODEL_PATH.name}")
    print(f"  特徴量数: {len(model.feature_name())}")

    df = load_test_data()
    evaluate_trifecta(
        model,
        df,
        encoders,
        "テスト / 1着1号艇以外予想",
        EVALUATION_CSV_PATH,
        PREDICTIONS_CSV_PATH,
    )

    print("\nSHAP分析")
    x_all = prepare_all(df, encoders)
    x_sample = x_all.sample(min(SHAP_SAMPLE_SIZE, len(x_all)), random_state=42)
    run_shap_analysis(
        model,
        x_sample,
        SHAP_BEESWARM_PNG_PATH,
        SHAP_WATERFALL_PNG_PATH,
        title="1着1号艇以外予想モデル（テスト）",
    )

    print(f"\n保存完了:")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {SHAP_BEESWARM_PNG_PATH}")
    print(f"  {SHAP_WATERFALL_PNG_PATH}")


if __name__ == "__main__":
    main()
