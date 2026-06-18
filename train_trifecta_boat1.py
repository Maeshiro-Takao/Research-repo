"""
1着1号艇予想モデル（LightGBM Ranker）

- 対象: 実際の1着=1号艇のレースのみ
- 学習: 1号艇を1着固定し、2〜6号艇の5艇でランキング学習
- 目的: 1号艇1着前提で2着・3着（3連単 1-◯-◯）を予測
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from trifecta_training_utils import (
    BASE_LGBM_RANKER_PARAMS,
    CATEGORICAL_FEATURES,
    RECENCY_HALF_LIFE_YEARS,
    VALIDATION_YEARS,
    build_feature_list,
    compute_recency_weights,
    encode_categorical_columns,
    run_shap_analysis,
    setup_japanese_font,
    temporal_train_valid_split,
    train_incremental_by_year,
)

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "レースデータ" / "丸亀学習用_レースデータ.csv"
MODEL_DIR = BASE_DIR / "models" / "1着1号艇予想"
TRAIN_OUTPUT_DIR = MODEL_DIR / "学習"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgbm_trifecta_boat1_model.txt"
ENCODER_PATH = MODEL_DIR / "trifecta_boat1_encoders.pkl"
OPTUNA_SUMMARY_PATH = TRAIN_OUTPUT_DIR / "最適化結果.json"
PREDICTIONS_CSV_PATH = TRAIN_OUTPUT_DIR / "予測結果.csv"
EVALUATION_CSV_PATH = TRAIN_OUTPUT_DIR / "評価結果.csv"
VALID_EVALUATION_CSV_PATH = TRAIN_OUTPUT_DIR / "検証_評価結果.csv"
SHAP_BEESWARM_PNG_PATH = TRAIN_OUTPUT_DIR / "特徴量影響方向.png"
SHAP_WATERFALL_PNG_PATH = TRAIN_OUTPUT_DIR / "特徴量ウォーターフォール.png"

RACE_KEY = ["開催日", "日目", "レース"]
TARGET_BOAT = 1
TOP_N_LIST = [1, 3, 5, 10, 30]
SHAP_SAMPLE_SIZE = 2000
N_TRIALS = 50

FEATURES: list[str] = []


def configure_features(df: pd.DataFrame) -> None:
    global FEATURES
    FEATURES = build_feature_list(df)


def load_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}")
    df = pd.read_csv(RACE_DATA_PATH)
    configure_features(df)
    df = filter_target_races(df)

    matched = df["級"].notna().sum() if "級" in df.columns else 0
    print(f"  対象: {len(df)} 行 / {len(df) // 6} レース（1着=1号艇、学習は2〜6号艇）")
    print(f"  選手データ結合: {matched} 行 ({matched / len(df):.1%})")
    print(f"  特徴量数: {len(FEATURES)}")
    return df


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
    """2〜6号艇のみを対象にランキング学習用データを作成する"""
    df = filter_complete_races(df)
    df = df.sort_values(RACE_KEY).reset_index(drop=True)
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    df = exclude_boat1(df)
    boat_numbers = df["艇"].astype(int).values
    groups = df.groupby(RACE_KEY, sort=False).size().tolist()
    df = df.drop(columns=["日目", "選手名"], errors="ignore")
    encoders = encoders or {}
    df, encoders = encode_categorical_columns(df, encoders, update_encoders=True)
    x = df[FEATURES].copy()
    x.columns = FEATURES
    y = 7 - df["着"].astype(int)
    weights = compute_recency_weights(df["開催日"])
    return x, y, groups, encoders, boat_numbers, weights


def build_rank_dataset(df: pd.DataFrame, encoders: dict):
    x, y, groups, _, _, weights = prepare(df, encoders)
    cat = [c for c in CATEGORICAL_FEATURES if c in FEATURES]
    train_set = lgb.Dataset(
        x, label=y, weight=weights, group=groups,
        categorical_feature=cat, free_raw_data=False,
    )
    return train_set, x


def suggest_lgbm_params(trial: optuna.Trial) -> tuple[dict, int]:
    params = {
        **BASE_LGBM_RANKER_PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }
    return params, trial.suggest_int("num_boost_round", 100, 500)


def create_objective(train_set: lgb.Dataset, valid_set: lgb.Dataset):
    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = suggest_lgbm_params(trial)
        model = lgb.train(
            params, train_set, num_boost_round=num_boost_round,
            valid_sets=[valid_set], valid_names=["valid"],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)],
        )
        return model.best_score["valid"]["ndcg@1"]
    return objective


def count_races(df: pd.DataFrame) -> int:
    return df.drop_duplicates(RACE_KEY).shape[0]


def optimize_model(df: pd.DataFrame, encoders: dict | None):
    train_df, valid_df = temporal_train_valid_split(df)
    n_train = count_races(train_df)
    n_valid = count_races(valid_df)
    if n_train < 100:
        raise ValueError(f"学習レース数が少なすぎます: {n_train}")
    if n_valid < 20:
        raise ValueError(f"検証レース数が少なすぎます: {n_valid}")

    print(f"  アルゴリズム: LightGBM Ranker (lambdarank)")
    print(f"  時系列分割: 学習 {n_train} レース / 検証 {n_valid} レース（直近{VALIDATION_YEARS}年）")
    print(f"  重み付け: 半減期 {RECENCY_HALF_LIFE_YEARS} 年（新しいデータほど重視）")

    _, _, _, encoders, _, _ = prepare(train_df, encoders or {})
    train_set, _ = build_rank_dataset(train_df, encoders)
    valid_set, _ = build_rank_dataset(valid_df, encoders)

    print(f"  Optuna 最適化開始（{N_TRIALS} trials, 検証 ndcg@1）...")
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(train_set, valid_set), n_trials=N_TRIALS)
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best valid ndcg@1: {study.best_value:.6f}")

    model, encoders, x_train, train_meta = train_incremental_by_year(
        df, study.best_params, BASE_LGBM_RANKER_PARAMS, prepare, build_rank_dataset,
    )

    summary = {
        "model": "1着1号艇予想",
        "target_races": "1着=1号艇",
        "ranking_boats": "2〜6号艇（5艇）",
        "n_races": count_races(df),
        "n_train_races": n_train,
        "n_valid_races": n_valid,
        "best_value": study.best_value,
        "best_metric": "valid ndcg@1",
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "validation_years": VALIDATION_YEARS,
        "best_params": study.best_params,
        "n_trials": N_TRIALS,
        "features": FEATURES,
        **train_meta,
    }
    return model, summary, x_train, encoders, valid_df


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

    x, _, _, _, boat_numbers, _ = prepare(race_df, encoders)
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
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    df = load_data()
    print(
        f"  期間: {pd.to_datetime(df['開催日']).min().date()} 〜 "
        f"{pd.to_datetime(df['開催日']).max().date()}"
    )

    _, _, _, encoders, _, _ = prepare(df)
    model, summary, x_train, encoders, valid_df = optimize_model(df, encoders)

    evaluate_trifecta(model, df, encoders, "学習データ全体", EVALUATION_CSV_PATH)
    evaluate_trifecta(
        model, valid_df, encoders,
        f"直近{VALIDATION_YEARS}年ホールドアウト",
        VALID_EVALUATION_CSV_PATH,
    )

    model.save_model(str(MODEL_PATH))
    joblib.dump(encoders, ENCODER_PATH)
    with open(OPTUNA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_predictions(model, df, encoders, PREDICTIONS_CSV_PATH)

    print("\nSHAP分析")
    x_sample = x_train.sample(min(SHAP_SAMPLE_SIZE, len(x_train)), random_state=42)
    run_shap_analysis(
        model, x_sample,
        SHAP_BEESWARM_PNG_PATH, SHAP_WATERFALL_PNG_PATH,
        title="1着1号艇予想モデル",
    )

    print(f"\n保存完了:")
    print(f"  {MODEL_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"  {OPTUNA_SUMMARY_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {SHAP_BEESWARM_PNG_PATH}")
    print(f"  {SHAP_WATERFALL_PNG_PATH}")


if __name__ == "__main__":
    main()
