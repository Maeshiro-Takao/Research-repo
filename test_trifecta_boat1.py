"""
1号艇が1着のレースを対象に、学習済みモデルで3連単（1-◯-◯）を予測・評価する。

- 推論: 1号艇を1着固定し、2〜6号艇のランキングから2着・3着を予測
- 前処理・推論・評価: train_trifecta_boat1.py と同じロジック
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
MODEL_DIR = BASE_DIR / "models" / "1着1号艇予想"
TEST_OUTPUT_DIR = MODEL_DIR / "テスト"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgbm_trifecta_boat1_model.txt"
ENCODER_PATH = MODEL_DIR / "trifecta_boat1_encoders.pkl"
PREDICTIONS_CSV_PATH = TEST_OUTPUT_DIR / "予測結果.csv"
EVALUATION_CSV_PATH = TEST_OUTPUT_DIR / "評価結果.csv"
SHAP_BEESWARM_PNG_PATH = TEST_OUTPUT_DIR / "特徴量影響方向.png"
SHAP_WATERFALL_PNG_PATH = TEST_OUTPUT_DIR / "特徴量ウォーターフォール.png"

RACE_KEY = ["開催日", "日目", "レース"]
TARGET_BOAT = 1
TOP_N_LIST = [1, 3, 5, 10, 30]
SHAP_SAMPLE_SIZE = 2000

FEATURES: list[str] = []


def configure_features(df: pd.DataFrame) -> None:
    global FEATURES
    FEATURES = build_feature_list(df)


def load_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}")
    df = pd.read_csv(RACE_DATA_PATH, low_memory=False)
    configure_features(df)
    df = filter_target_races(df)

    matched = df["級"].notna().sum() if "級" in df.columns else 0
    print(f"  対象: {len(df)} 行 / {len(df) // 6} レース（1着=1号艇、推論は2〜6号艇）")
    print(f"  選手データ結合: {matched} 行 ({matched / len(df):.1%})")
    print(f"  特徴量数: {len(FEATURES)}")
    return df


def load_model_and_encoder() -> tuple[lgb.Booster, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {ENCODER_PATH}")
    return lgb.Booster(model_file=str(MODEL_PATH)), joblib.load(ENCODER_PATH)


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    return df.loc[sizes == 6].copy()


def filter_target_races(df: pd.DataFrame) -> pd.DataFrame:
    """1着が1号艇のレースのみ残す"""
    df = filter_complete_races(df)
    parts: list[pd.DataFrame] = []
    for _, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        if int(race_df.loc[race_df["着"] == 1, "艇"].iloc[0]) == 1:
            parts.append(race_df)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def exclude_boat1(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["艇"].astype(int) != TARGET_BOAT].copy()


def prepare(df: pd.DataFrame, encoders=None):
    """2〜6号艇のみを対象に推論用データを作成する"""
    df = filter_complete_races(df)
    df = df.sort_values(RACE_KEY).reset_index(drop=True)
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    df = exclude_boat1(df)
    boat_numbers = df["艇"].astype(int).values
    df = df.drop(columns=["日目", "選手名"], errors="ignore")
    encoders = encoders or {}
    for col in CATEGORICAL_FEATURES:
        if col not in encoders:
            raise KeyError(
                f"エンコーダに {col} がありません。"
                "先に train_trifecta_boat1.py を実行してください。"
            )
    df, encoders = encode_categorical_columns(df, encoders, update_encoders=False)
    x = df[FEATURES].copy()
    x.columns = FEATURES
    return x, boat_numbers


def get_actual_trifecta(race_df: pd.DataFrame) -> tuple[int, int, int]:
    top3 = race_df.sort_values("着").head(3)
    return tuple(top3["艇"].astype(int).tolist())  # type: ignore[return-value]


def calc_trifecta_probs_boat1_fixed(
    boat_scores: dict[int, float],
) -> dict[tuple[int, int, int], float]:
    """
    1号艇を1着固定し、2〜6号艇のスコアから3連単確率を算出する。
    出力は必ず (1, j, k) 形式（j, k は 2〜6号艇）。
    """
    boats = sorted(boat_scores)
    if len(boats) != 5:
        raise ValueError(f"2〜6号艇のスコアが5艇分必要です: {len(boats)}艇")

    max_score = max(boat_scores.values())
    exp_scores = {b: np.exp(boat_scores[b] - max_score) for b in boats}
    total = sum(exp_scores.values())
    results: dict[tuple[int, int, int], float] = {}

    for second in boats:
        p2 = max(exp_scores[second] / total, 1e-12)
        remaining = [b for b in boats if b != second]
        rem_total = sum(exp_scores[b] for b in remaining)
        for third in remaining:
            p3 = max(exp_scores[third] / rem_total, 1e-12)
            results[(TARGET_BOAT, second, third)] = p2 * p3

    return results


def predict_race_trifecta(model, race_df, encoders):
    """1号艇を1着固定し、2〜6号艇の順位から3連単を予測する"""
    race_df = filter_complete_races(race_df)
    if len(race_df) != 6:
        raise ValueError("6艇揃ったレースが必要です")

    x, boat_numbers = prepare(race_df, encoders)
    scores = model.predict(x)
    boat_scores = {int(boat): float(score) for boat, score in zip(boat_numbers, scores)}

    ranking = sorted(
        calc_trifecta_probs_boat1_fixed(boat_scores).items(),
        key=lambda x: x[1],
        reverse=True,
    )
    pred_top = ranking[0][0]
    assert pred_top[0] == TARGET_BOAT, f"1号艇以外が1着になりました: {pred_top}"
    return pred_top, ranking


def evaluate_trifecta(model, df, encoders, label, output_path: Path):
    hits = {n: 0 for n in TOP_N_LIST}
    first_hits = second_hits = 0
    logloss_sum = 0.0
    n_races = 0
    for _, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        actual = get_actual_trifecta(race_df)
        pred_top, ranking = predict_race_trifecta(model, race_df, encoders)
        prob_map = dict(ranking)
        for n in TOP_N_LIST:
            if actual in [c for c, _ in ranking[:n]]:
                hits[n] += 1
        first_hits += pred_top[0] == actual[0]  # 予測1着は常に1号艇
        second_hits += pred_top[:2] == actual[:2]
        logloss_sum += -np.log(max(prob_map.get(actual, 1e-12), 1e-12))
        n_races += 1

    metrics = {
        "label": label,
        "n_races": n_races,
        "first_hit_rate": first_hits / n_races,
        "exacta_hit_rate": second_hits / n_races,
        "trifecta_top1_rate": hits[1] / n_races,
        "trifecta_avg_neg_log_prob": logloss_sum / n_races,
        **{f"trifecta_top{n}_rate": hits[n] / n_races for n in TOP_N_LIST},
    }

    print(f"\n=== 3連単評価 ({label}) ===")
    print(f"  レース数: {n_races}")
    for n in TOP_N_LIST:
        print(f"  3連単 TOP{n} 的中率: {hits[n] / n_races:.2%}")
    print(f"  1着艇番 的中率: {first_hits / n_races:.2%}")
    print(f"  2連単 的中率:   {second_hits / n_races:.2%}")
    print(f"  3連単 的中率:   {hits[1] / n_races:.2%}")
    print(f"  平均-log確率: {logloss_sum / n_races:.4f}")

    pd.DataFrame([metrics]).to_csv(output_path, index=False, encoding="UTF-8-sig")
    print(f"  評価結果CSV: {output_path}")
    return metrics


def save_predictions(model, df, encoders, output_path):
    rows = []
    for key, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        actual = get_actual_trifecta(race_df)
        pred_top, ranking = predict_race_trifecta(model, race_df, encoders)
        rows.append({
            "開催日": key[0], "レース": key[2],
            "実際3連単": "-".join(map(str, actual)),
            "予測3連単": "-".join(map(str, pred_top)),
            "予測確率": ranking[0][1],
            "的中": pred_top == actual,
            "TOP10": " / ".join(
                f"{'-'.join(map(str, c))}({p:.4f})" for c, p in ranking[:10]
            ),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="UTF-8-sig")
    print(f"  予測結果CSV: {output_path}")


def main():
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    model, encoders = load_model_and_encoder()
    print(f"  モデル: {MODEL_PATH.name}")
    print(f"  特徴量数: {len(model.feature_name())}")

    df = load_data()
    print(
        f"  期間: {pd.to_datetime(df['開催日']).min().date()} 〜 "
        f"{pd.to_datetime(df['開催日']).max().date()}"
    )

    evaluate_trifecta(
        model, df, encoders, "テスト / 1着1号艇予想", EVALUATION_CSV_PATH
    )
    save_predictions(model, df, encoders, PREDICTIONS_CSV_PATH)

    print("\nSHAP分析")
    x_all, _ = prepare(df, encoders)
    x_sample = x_all.sample(min(SHAP_SAMPLE_SIZE, len(x_all)), random_state=42)
    run_shap_analysis(
        model,
        x_sample,
        SHAP_BEESWARM_PNG_PATH,
        SHAP_WATERFALL_PNG_PATH,
        title="1着1号艇予想モデル（テスト）",
    )

    print(f"\n保存完了:")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {SHAP_BEESWARM_PNG_PATH}")
    print(f"  {SHAP_WATERFALL_PNG_PATH}")


if __name__ == "__main__":
    main()
