from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import optuna
import pandas as pd
import shap

from trifecta_training_utils import (
    RECENCY_HALF_LIFE_YEARS,
    ROLLING_FEATURE_EXCLUDE,
    VALIDATION_YEARS,
    compute_recency_weights,
    temporal_train_valid_split,
    train_incremental_by_year,
)

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "レースデータ" / "丸亀学習用_レースデータ.csv"
OUTPUT_DIR = BASE_DIR / "models" / "1着1号艇以外予想"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "lgbm_trifecta_not_boat1_model.txt"
ENCODER_PATH = OUTPUT_DIR / "trifecta_not_boat1_encoders.pkl"
OPTUNA_SUMMARY_PATH = OUTPUT_DIR / "最適化結果.json"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "学習_予測結果.csv"
EVALUATION_CSV_PATH = OUTPUT_DIR / "学習_評価結果.csv"
SHAP_IMPORTANCE_PNG_PATH = OUTPUT_DIR / "特徴量重要度.png"
SHAP_BEESWARM_PNG_PATH = OUTPUT_DIR / "特徴量影響方向.png"
SHAP_IMPORTANCE_CSV_PATH = OUTPUT_DIR / "特徴量重要度.csv"

RACE_KEY = ["開催日", "日目", "レース"]
RACE_ROW_KEY = ["開催日", "日目", "レース", "艇"]

RACE_EXCLUDE_COLUMNS = {"レース", "着", "選手名", "日目", "開催日", "登番", "モーター", "ボート", "艇", "3連単オッズ"}
PLAYER_META_COLS = ["算出期間自", "算出期間至"]
RACE_BASE_FEATURES = [
    "展示",
    "展示順位", "展示差", "風速", "波高", "天気", "風向",
    "当地勝率",
]
TOP_N_LIST = [1, 3, 5, 10, 30]
SHAP_SAMPLE_SIZE = 2000
N_TRIALS = 50

FEATURES: list[str] = []
CATEGORICAL = ["天気", "風向", "級"]

BASE_LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "verbosity": -1,
    "seed": 42,
    "feature_pre_filter": False,
}

JAPANESE_FONT_CANDIDATES = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Hiragino Maru Gothic ProN",
    "Yu Gothic",
    "YuGothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAGothic",
    "MS Gothic",
]


def configure_features(df: pd.DataFrame) -> None:
    global FEATURES
    base = [c for c in RACE_BASE_FEATURES if c in df.columns]
    exclude = (
        RACE_EXCLUDE_COLUMNS
        | set(PLAYER_META_COLS)
        | set(RACE_BASE_FEATURES)
        | ROLLING_FEATURE_EXCLUDE
        | {"登番"}
    )
    player_features = [c for c in df.columns if c not in exclude]
    FEATURES = list(dict.fromkeys(base + player_features))


def setup_japanese_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in JAPANESE_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    for font in font_manager.fontManager.ttflist:
        if any(k in font.name for k in ("Hiragino", "Noto Sans CJK", "Yu Gothic", "Meiryo")):
            plt.rcParams["font.family"] = font.name
            plt.rcParams["axes.unicode_minus"] = False
            return font.name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def load_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}")
    df = pd.read_csv(RACE_DATA_PATH)
    configure_features(df)
    df = filter_target_races(df)

    matched = df["級"].notna().sum() if "級" in df.columns else 0
    print(f"  対象: {len(df)} 行 / {len(df) // 6} レース（1着=1号艇以外）")
    print(f"  選手データ結合: {matched} 行 ({matched / len(df):.1%})")
    print(f"  特徴量数: {len(FEATURES)}")
    return df


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    return df.loc[sizes == 6].copy()


def filter_target_races(df: pd.DataFrame) -> pd.DataFrame:
    """1着が1号艇以外のレースのみ残す"""
    df = filter_complete_races(df)
    parts: list[pd.DataFrame] = []
    for _, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        if int(race_df.loc[race_df["着"] == 1, "艇"].iloc[0]) != 1:
            parts.append(race_df)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def prepare(df: pd.DataFrame, encoders=None):
    df = filter_complete_races(df)
    df = df.sort_values(RACE_KEY).reset_index(drop=True)
    boat_numbers = df["艇"].astype(int).values
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    groups = df.groupby(RACE_KEY, sort=False).size().tolist()
    df = df.drop(columns=["日目", "選手名"], errors="ignore")
    encoders = encoders or {}
    for col in CATEGORICAL:
        if col not in encoders:
            encoders[col] = {v: i for i, v in enumerate(df[col].astype(str).unique())}
        df[col] = df[col].astype(str).map(encoders[col]).fillna(-1).astype(int)
    x = df[FEATURES]
    y = 7 - df["着"].astype(int)
    weights = compute_recency_weights(df["開催日"])
    return x, y, groups, encoders, boat_numbers, weights


def build_rank_dataset(df: pd.DataFrame, encoders: dict):
    x, y, groups, _, _, weights = prepare(df, encoders)
    cat = [c for c in CATEGORICAL if c in FEATURES]
    train_set = lgb.Dataset(
        x, label=y, weight=weights, group=groups,
        categorical_feature=cat, free_raw_data=False,
    )
    return train_set, x


def suggest_lgbm_params(trial: optuna.Trial) -> tuple[dict, int]:
    params = {
        **BASE_LGBM_PARAMS,
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


def train_with_params(train_set, best_params: dict) -> lgb.Booster:
    params = {**BASE_LGBM_PARAMS}
    for key in (
        "num_leaves", "max_depth", "learning_rate", "min_data_in_leaf",
        "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2",
    ):
        params[key] = best_params[key]
    return lgb.train(
        params, train_set, num_boost_round=best_params["num_boost_round"],
        callbacks=[lgb.log_evaluation(50)],
    )


def optimize_model(df: pd.DataFrame, encoders: dict | None):
    train_df, valid_df = temporal_train_valid_split(df)
    n_train = len(train_df) // 6
    n_valid = len(valid_df) // 6
    if n_train < 100:
        raise ValueError(f"学習レース数が少なすぎます: {n_train}")
    if n_valid < 20:
        raise ValueError(f"検証レース数が少なすぎます: {n_valid}")

    print(f"  時系列分割: 学習 {n_train} レース / 検証 {n_valid} レース（直近{VALIDATION_YEARS}年）")
    print(f"  重み付け: 半減期 {RECENCY_HALF_LIFE_YEARS} 年（新しいデータほど重視）")
    print(f"  除外特徴量: 直近10年/5年の勝率・連帯率")

    _, _, _, encoders, _, _ = prepare(train_df, encoders or {})
    train_set, _ = build_rank_dataset(train_df, encoders)
    valid_set, _ = build_rank_dataset(valid_df, encoders)

    print(f"  Optuna 最適化開始（{N_TRIALS} trials, 検証 ndcg@1）...")
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(train_set, valid_set), n_trials=N_TRIALS)
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best valid ndcg@1: {study.best_value:.6f}")

    model, encoders, x_train, train_meta = train_incremental_by_year(
        df, study.best_params, BASE_LGBM_PARAMS, prepare, build_rank_dataset,
    )

    summary = {
        "model": "1着1号艇以外予想",
        "n_races": len(df) // 6,
        "n_train_races": n_train,
        "n_valid_races": n_valid,
        "best_value": study.best_value,
        "best_metric": "valid ndcg@1",
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "validation_years": VALIDATION_YEARS,
        "excluded_features": sorted(ROLLING_FEATURE_EXCLUDE),
        "best_params": study.best_params,
        "n_trials": N_TRIALS,
        "features": FEATURES,
        **train_meta,
    }
    return model, summary, x_train, encoders, valid_df


def get_actual_trifecta(race_df: pd.DataFrame) -> tuple[int, int, int]:
    top3 = race_df.sort_values("着").head(3)
    return tuple(top3["艇"].astype(int).tolist())  # type: ignore[return-value]


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


def predict_race_trifecta(model, race_df, encoders):
    x, _, _, _, boat_numbers, _ = prepare(race_df, encoders)
    scores = model.predict(x)
    score_arr = np.zeros(6)
    for idx, boat in enumerate(boat_numbers):
        score_arr[boat - 1] = scores[idx]
    ranking = sorted(
        calc_trifecta_probs_from_scores(score_arr).items(),
        key=lambda x: x[1], reverse=True,
    )
    return ranking[0][0], ranking


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
        first_hits += pred_top[0] == actual[0]
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


def run_shap_analysis(model, x_sample):
    shap_values = shap.TreeExplainer(model).shap_values(x_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_sample, plot_type="bar", show=False)
    plt.title("特徴量", fontsize=14)
    plt.tight_layout()
    plt.savefig(SHAP_IMPORTANCE_PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, x_sample, show=False)
    plt.title("特徴量", fontsize=14)
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame({
        "特徴量": x_sample.columns,
        "重要度": np.abs(shap_values).mean(axis=0),
    }).sort_values("重要度", ascending=False).to_csv(
        SHAP_IMPORTANCE_CSV_PATH, index=False, encoding="UTF-8-sig"
    )

    print(f"  SHAP PNG: {SHAP_IMPORTANCE_PNG_PATH}")
    print(f"  SHAP PNG: {SHAP_BEESWARM_PNG_PATH}")
    print(f"  SHAP CSV: {SHAP_IMPORTANCE_CSV_PATH}")


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

    evaluate_trifecta(
        model, df, encoders, "学習データ全体", EVALUATION_CSV_PATH
    )
    holdout_path = OUTPUT_DIR / "検証_評価結果.csv"
    evaluate_trifecta(
        model, valid_df, encoders, f"直近{VALIDATION_YEARS}年ホールドアウト", holdout_path
    )

    model.save_model(str(MODEL_PATH))
    joblib.dump(encoders, ENCODER_PATH)
    with open(OPTUNA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_predictions(model, df, encoders, PREDICTIONS_CSV_PATH)

    print("\nSHAP分析")
    x_sample = x_train.sample(min(SHAP_SAMPLE_SIZE, len(x_train)), random_state=42)
    run_shap_analysis(model, x_sample)

    print(f"\n保存完了:")
    print(f"  {MODEL_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"  {OPTUNA_SUMMARY_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {SHAP_IMPORTANCE_PNG_PATH}")
    print(f"  {SHAP_BEESWARM_PNG_PATH}")
    print(f"  {SHAP_IMPORTANCE_CSV_PATH}")


if __name__ == "__main__":
    main()
