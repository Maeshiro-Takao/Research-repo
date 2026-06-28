"""
1レース6艇 LightGBM Ranker + Plackett–Luce 3連単予想

パイプライン:
  6艇ランキング学習（LightGBM Ranker, group=6, ラベル=着順）
  → 各艇スコア推論
  → Plackett–Luce で120通り3連単確率

- データ: レース/選手/気象 CSV（2014〜2024年）
- Optuna + Walk Forward 評価
- 保存先: models/3連単予想/（既存 models/レース着順ランキング/ とは別）
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap

from plackett_luce import (
    combo_to_str,
    compute_ndcg_at_k,
    predict_trifecta_from_scores,
    prob_sum,
)
from trifecta_training_utils import (
    BASE_LGBM_RANKER_PARAMS,
    RECENCY_HALF_LIFE_YEARS,
    UNKNOWN_CATEGORY_TOKEN,
    build_lgbm_params,
    compute_recency_weights,
    get_actual_trifecta,
    get_calendar_years,
    run_shap_analysis,
    setup_japanese_font,
)

# ---------------------------------------------------------------------------
# パス・定数
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_DIR = BASE_DIR / "レースデータ"
PLAYER_DATA_DIR = BASE_DIR / "選手データ"
WEATHER_DATA_DIR = BASE_DIR / "気象データ"

TRAIN_YEARS = range(2014, 2025)
TEST_YEARS = range(2025, 2026)

MODEL_DIR = BASE_DIR / "models" / "3連単予想"
OUTPUT_DIR = MODEL_DIR / "学習"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgbm_race_ranker.txt"
ENCODER_PATH = MODEL_DIR / "race_ranker_encoders.pkl"
CONFIG_PATH = MODEL_DIR / "race_ranker_config.json"
WALK_FORWARD_CSV_PATH = OUTPUT_DIR / "WalkForward評価.csv"
LEARNING_CURVE_PNG = OUTPUT_DIR / "学習曲線.png"
EVALUATION_CSV_PATH = OUTPUT_DIR / "評価結果.csv"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "予測結果.csv"
PREDICTIONS_DETAIL_CSV_PATH = OUTPUT_DIR / "予測結果_120通り.csv"
SHAP_BEESWARM_PNG = OUTPUT_DIR / "特徴量影響方向.png"
SHAP_WATERFALL_PNG = OUTPUT_DIR / "特徴量ウォーターフォール.png"
SHAP_IMPORTANCE_PNG = OUTPUT_DIR / "特徴量重要度.png"
SHAP_IMPORTANCE_CSV = OUTPUT_DIR / "特徴量重要度.csv"
TEST_OUTPUT_DIR = MODEL_DIR / "テスト"

RACE_KEY = ["開催日", "日目", "レース"]
TOP_N_LIST = [1, 5, 10, 20]
RACE_ROW_KEY = RACE_KEY + ["艇"]
MERGED_COLS = (
    RACE_ROW_KEY
    + ["登番", "選手名", "着", "モーター", "ボート", "展示", "3連単オッズ"]
    + ["天気", "風向", "風速", "波高"]
    + ["級", "身長", "体重"]
    + ["勝率", "2連率", "3連率", "当地勝率", "当地2連率", "当地3連率"]
    + ["モーター勝率", "モーター2連率", "モーター3連率"]
    + ["ボート勝率", "ボート2連率", "ボート3連率"]
    + ["平均スタートタイミング"]
    + [f"{c}コース進入回数" for c in range(1, 7)]
    + [f"{c}コース複勝率" for c in range(1, 7)]
    + [f"{c}コース平均スタートタイミング" for c in range(1, 7)]
    + [f"{c}コース平均スタート順位" for c in range(1, 7)]
    + ["算出期間自", "算出期間至"]
)

REFERENCE_DATE = pd.Timestamp("2014-01-01")
# 特徴量に含めない列（キー・ラベル・メタ情報）
EXCLUDE_FROM_FEATURES = {"選手名", "レース", "開催日", "日目", "算出期間自", "算出期間至", "着", "3連単オッズ", "艇"}
CATEGORICAL_COLS = ["登番", "モーター", "ボート", "級", "天気", "風向"]
RELATIVE_BASE_COLS = [
    "展示", "勝率", "2連率", "3連率", "当地勝率", "当地2連率", "当地3連率",
    "モーター勝率", "モーター2連率", "モーター3連率",
    "ボート勝率", "ボート2連率", "ボート3連率", "平均スタートタイミング",
    "1コース複勝率", "2コース複勝率", "3コース複勝率",
    "4コース複勝率", "5コース複勝率", "6コース複勝率",
]
COURSE_STAT_COLS = [
    f"{c}コース進入回数" for c in range(1, 7)
] + [
    f"{c}コース複勝率" for c in range(1, 7)
] + [
    f"{c}コース平均スタートタイミング" for c in range(1, 7)
] + [
    f"{c}コース平均スタート順位" for c in range(1, 7)
]
PLAYER_NUMERIC_COLS = [
    "体重", "勝率", "2連率", "3連率", "当地勝率", "当地2連率", "当地3連率",
    "モーター勝率", "モーター2連率", "モーター3連率",
    "ボート勝率", "ボート2連率", "ボート3連率",
    "平均スタートタイミング", *COURSE_STAT_COLS,
]
NUMERIC_COLS = (
    ["開催経過日", "展示", "風速", "波高"]
    + PLAYER_NUMERIC_COLS
    + [f"{c}_レース内順位" for c in RELATIVE_BASE_COLS]
    + [f"{c}_レース内差" for c in RELATIVE_BASE_COLS]
)
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

NDCG_METRICS = ["ndcg@1", "ndcg@3"]
N_TRIALS = 25
SHAP_SAMPLE_SIZE = 2000
EARLY_STOPPING_ROUNDS = 50
MIN_BOOST_ROUND = 50
WF_MIN_TRAIN_YEARS = 7


def build_feature_names() -> list[str]:
    return FEATURES


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------
def _merge_year(year: int) -> pd.DataFrame:
    race_path = RACE_DATA_DIR / f"丸亀_{year}_レースデータ.csv"
    player_path = PLAYER_DATA_DIR / f"丸亀_{year}_選手データ.csv"
    weather_path = WEATHER_DATA_DIR / f"丸亀_{year}_気象データ.csv"
    if not (race_path.exists() and player_path.exists() and weather_path.exists()):
        raise FileNotFoundError(f"{year}年の CSV が不足しています")
    race_df = pd.read_csv(race_path, low_memory=False)
    player_df = pd.read_csv(player_path, low_memory=False)
    weather_df = pd.read_csv(weather_path, low_memory=False)
    merged = race_df.merge(
        player_df, on=RACE_ROW_KEY + ["登番"], how="left", validate="one_to_one",
    )
    merged = merged.merge(weather_df, on=RACE_KEY, how="left", validate="many_to_one")
    return merged[[c for c in MERGED_COLS if c in merged.columns]]


def load_race_data(years: range | list[int]) -> pd.DataFrame:
    parts = [_merge_year(y) for y in years]
    return pd.concat(parts, ignore_index=True)


def load_training_data() -> pd.DataFrame:
    print("データ読み込み: 2014〜2024年（レース/選手/気象）")
    df = load_race_data(TRAIN_YEARS)
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    df = df.loc[sizes == 6].copy()
    df["__year"] = pd.to_datetime(df["開催日"]).dt.year
    print(f"  行数: {len(df)} / レース数: {len(df) // 6} / 特徴量数: {len(FEATURES)}")
    return df


def subset_years(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    return df.loc[df["__year"].isin(years)].copy()


# ---------------------------------------------------------------------------
# 特徴量前処理（ベクトル化）
# ---------------------------------------------------------------------------
def _add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(RACE_KEY + ["艇"]).copy()
    work["開催経過日"] = (pd.to_datetime(work["開催日"]) - REFERENCE_DATE).dt.days
    for col in RELATIVE_BASE_COLS:
        rank_col = f"{col}_レース内順位"
        diff_col = f"{col}_レース内差"
        if col not in work.columns:
            work[rank_col] = 0.0
            work[diff_col] = 0.0
            continue
        vals = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
        keys = [work[key] for key in RACE_KEY]
        work[rank_col] = vals.groupby(keys).rank(method="min")
        work[diff_col] = vals - vals.groupby(keys).transform("mean")
    return work


def _encode_categoricals(frame: pd.DataFrame, encoders: dict, *, fit: bool) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    for col in CATEGORICAL_COLS:
        if col not in frame.columns:
            continue
        encoders.setdefault(col, {})
        values = frame[col].astype(str)
        if fit:
            next_idx = max(encoders[col].values(), default=-1) + 1
            for value in values.unique():
                if value not in encoders[col]:
                    encoders[col][value] = next_idx
                    next_idx += 1
        mapped = values.map(encoders[col])
        if mapped.isna().any():
            if UNKNOWN_CATEGORY_TOKEN not in encoders[col]:
                encoders[col][UNKNOWN_CATEGORY_TOKEN] = max(encoders[col].values(), default=-1) + 1
            mapped = mapped.fillna(encoders[col][UNKNOWN_CATEGORY_TOKEN])
        frame[col] = mapped.astype(int)
    return frame, encoders


def prepare_dataset(
    df: pd.DataFrame,
    encoders: dict | None = None,
    *,
    fit_encoders: bool = True,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[int], dict, np.ndarray]:
    """1レース6行を group=6 のランキング学習用テンソルに変換"""
    feature_names = feature_names or FEATURES
    encoders = encoders or {}
    work = _add_relative_features(df)
    x = work.reindex(columns=feature_names, fill_value=0)
    for col in NUMERIC_COLS:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    x, encoders = _encode_categoricals(x, encoders, fit=fit_encoders)
    y = pd.Series(7 - work["着"].astype(int), dtype=int)
    groups = [6] * (len(work) // 6)
    weights = compute_recency_weights(work["開催日"])
    return x, y, groups, encoders, weights


def build_lgb_dataset(df: pd.DataFrame, encoders: dict, *, fit_encoders: bool) -> tuple[lgb.Dataset, dict]:
    x, y, groups, encoders, weights = prepare_dataset(df, encoders, fit_encoders=fit_encoders)
    return lgb.Dataset(x, label=y, weight=weights, group=groups, free_raw_data=False), encoders


def load_model() -> tuple[lgb.Booster, dict, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}\n先に train_race_ranker.py を実行してください。")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {ENCODER_PATH}")
    features = FEATURES
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            features = json.load(f).get("features", FEATURES)
    return lgb.Booster(model_file=str(MODEL_PATH)), joblib.load(ENCODER_PATH), features


def predict_boat_scores(
    model: lgb.Booster,
    df: pd.DataFrame,
    encoders: dict,
    *,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """全行のランキングスコアを一括推論"""
    feature_names = feature_names or FEATURES
    x, _, _, _, _ = prepare_dataset(df, encoders, fit_encoders=False, feature_names=feature_names)
    return model.predict(x)


def predict_race_trifecta(
    race_scores: np.ndarray,
    boats: list[int] | None = None,
) -> tuple[dict[tuple[int, int, int], float], list[tuple[tuple[int, int, int], float, int]]]:
    """1レース分のスコア → Plackett–Luce 120通り確率"""
    return predict_trifecta_from_scores(race_scores, boats=boats)


def evaluate_model(
    model: lgb.Booster,
    df: pd.DataFrame,
    encoders: dict,
    label: str,
    *,
    feature_names: list[str] | None = None,
    output_dir: Path | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """6艇ランキング + Plackett–Luce 3連単の総合評価"""
    feature_names = feature_names or FEATURES
    sorted_df = df.sort_values(RACE_KEY + ["艇"])
    scores = predict_boat_scores(model, sorted_df, encoders, feature_names=feature_names)

    hits = {n: 0 for n in TOP_N_LIST}
    ndcg1_sum = 0.0
    ndcg3_sum = 0.0
    n_races = 0
    race_rows: list[dict] = []
    detail_rows: list[dict] = []

    offset = 0
    for key, race_df in sorted_df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        race_scores = scores[offset : offset + 6]
        offset += 6
        boats = race_df["艇"].astype(int).tolist()
        relevance = (7 - race_df["着"].astype(int)).to_numpy()

        ndcg1 = compute_ndcg_at_k(race_scores, relevance, 1)
        ndcg3 = compute_ndcg_at_k(race_scores, relevance, 3)
        ndcg1_sum += ndcg1
        ndcg3_sum += ndcg3

        actual = get_actual_trifecta(race_df)
        prob_map, ranking = predict_race_trifecta(race_scores, boats=boats)
        total_prob = prob_sum(prob_map)
        top_combos = [combo for combo, _, _ in ranking]

        hit_rank = next((r for combo, _, r in ranking if combo == actual), None)
        actual_prob = prob_map.get(actual, 0.0)

        for n in TOP_N_LIST:
            if hit_rank is not None and hit_rank <= n:
                hits[n] += 1
        n_races += 1

        race_rows.append({
            "開催日": key[0],
            "日目": key[1],
            "レース": key[2],
            "実際3連単": combo_to_str(actual),
            "予測3連単": combo_to_str(top_combos[0]),
            "予測確率": ranking[0][1],
            "的中組み合わせ予測確率": actual_prob,
            "的中組み合わせ予測順位": hit_rank if hit_rank is not None else "圏外",
            "確率合計": total_prob,
            "NDCG@1": ndcg1,
            "NDCG@3": ndcg3,
        })

        for combo, prob, rank in ranking:
            detail_rows.append({
                "開催日": key[0],
                "日目": key[1],
                "レース": key[2],
                "3連単": combo_to_str(combo),
                "予測確率": prob,
                "予測順位": rank,
            })

    if n_races == 0:
        raise ValueError(f"評価対象レースがありません: {label}")

    metrics = {
        "label": label,
        "n_races": n_races,
        "ndcg_at_1": ndcg1_sum / n_races,
        "ndcg_at_3": ndcg3_sum / n_races,
        **{f"trifecta_top{n}_rate": hits[n] / n_races for n in TOP_N_LIST},
    }

    print(f"\n=== 評価 ({label}) ===")
    print(f"  レース数: {n_races}")
    print(f"  NDCG@1: {metrics['ndcg_at_1']:.6f}")
    print(f"  NDCG@3: {metrics['ndcg_at_3']:.6f}")
    for n in TOP_N_LIST:
        print(f"  3連単 TOP{n} 的中率: {metrics[f'trifecta_top{n}_rate']:.2%}")

    race_df_out = pd.DataFrame(race_rows)
    detail_df_out = pd.DataFrame(detail_rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_path = output_dir / "評価結果.csv"
        pred_path = output_dir / "予測結果.csv"
        detail_path = output_dir / "予測結果_120通り.csv"
        pd.DataFrame([metrics]).to_csv(eval_path, index=False, encoding="UTF-8-sig")
        race_df_out.to_csv(pred_path, index=False, encoding="UTF-8-sig")
        detail_df_out.to_csv(detail_path, index=False, encoding="UTF-8-sig")
        print(f"  評価結果CSV: {eval_path}")
        print(f"  予測結果CSV: {pred_path}")
        print(f"  120通りCSV: {detail_path}")

    return metrics, race_df_out, detail_df_out


def evaluate_trifecta(
    model: lgb.Booster,
    df: pd.DataFrame,
    encoders: dict,
    label: str,
    *,
    feature_names: list[str] | None = None,
    output_path: Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    """後方互換ラッパー"""
    output_dir = output_path.parent if output_path is not None else None
    metrics, race_df, _ = evaluate_model(
        model, df, encoders, label,
        feature_names=feature_names,
        output_dir=output_dir,
    )
    if output_path is not None and output_path.name != "評価結果.csv":
        pd.DataFrame([metrics]).to_csv(output_path, index=False, encoding="UTF-8-sig")
    return metrics, race_df


def save_shap_importance(
    model: lgb.Booster,
    x_sample: pd.DataFrame,
    png_path: Path,
    csv_path: Path,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    feature_names = [str(c) for c in x_sample.columns]
    importance = pd.DataFrame(
        {
            "特徴量": feature_names,
            "平均絶対SHAP": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("平均絶対SHAP", ascending=False)
    importance.to_csv(csv_path, index=False, encoding="UTF-8-sig")

    top = importance.head(25)
    plt.figure(figsize=(10, 8))
    plt.barh(top["特徴量"][::-1], top["平均絶対SHAP"][::-1])
    plt.xlabel("平均 |SHAP|")
    plt.title("特徴量重要度（SHAP）")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP重要度: {png_path}")
    print(f"  SHAP重要度CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Walk Forward / Optuna
# ---------------------------------------------------------------------------
def build_walk_forward_folds(df: pd.DataFrame) -> list[tuple[list[int], int]]:
    years = get_calendar_years(df)
    folds = [(years[:idx], years[idx]) for idx in range(WF_MIN_TRAIN_YEARS, len(years))]
    if not folds:
        raise ValueError("Walk Forward 分割を構築できません")
    return folds


def suggest_lgbm_params(trial: optuna.Trial) -> tuple[dict, int]:
    params = {
        **BASE_LGBM_RANKER_PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 16, 96),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 150),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
    }
    return params, trial.suggest_int("num_boost_round", 100, 400)


def run_optuna(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> optuna.Study:
    """最終年ホールドアウトで Optuna（Walk Forward は trial ごとに回さない）"""
    train_set, encoders = build_lgb_dataset(train_df, {}, fit_encoders=True)
    valid_set, _ = build_lgb_dataset(valid_df, encoders, fit_encoders=False)

    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = suggest_lgbm_params(trial)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.log_evaluation(0),
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            ],
        )
        return float(model.best_score["valid"]["ndcg@1"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    return study


def plot_learning_curves(fold_curves: list[tuple[str, dict]], png_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, metric in enumerate(NDCG_METRICS):
        for label, evals in fold_curves:
            curve = evals.get("valid", {}).get(metric, [])
            if curve:
                axes[0, col].plot(range(1, len(curve) + 1), curve, label=label)
        axes[0, col].set_title(f"Walk Forward 検証 {metric.upper()}")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(alpha=0.3)

        last_label, last_evals = fold_curves[-1]
        for split, jp in (("train", "学習"), ("valid", "検証")):
            curve = last_evals.get(split, {}).get(metric, [])
            if curve:
                axes[1, col].plot(range(1, len(curve) + 1), curve, label=jp)
        axes[1, col].set_title(f"最終フォールド {last_label} {metric.upper()}")
        axes[1, col].legend(fontsize=9)
        axes[1, col].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  学習曲線: {png_path}")


def run_walk_forward(
    df: pd.DataFrame,
    folds: list[tuple[list[int], int]],
    lgbm_params: dict,
    num_boost_round: int,
) -> tuple[list[dict], list[tuple[str, dict]], list[int]]:
    fold_metrics: list[dict] = []
    fold_curves: list[tuple[str, dict]] = []
    best_iters: list[int] = []

    print("\nWalk Forward 評価（OOS）")
    for train_years, valid_year in folds:
        label = f"学習{train_years[0]}-{train_years[-1]}→検証{valid_year}"
        train_set, encoders = build_lgb_dataset(subset_years(df, train_years), {}, fit_encoders=True)
        valid_set, _ = build_lgb_dataset(subset_years(df, [valid_year]), encoders, fit_encoders=False)

        evals: dict = {}
        model = lgb.train(
            lgbm_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.record_evaluation(evals),
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_iter = model.best_iteration or num_boost_round
        best_iters.append(best_iter)
        fold_curves.append((label, evals))

        print(f"\n  {label}  iteration={best_iter}")
        metrics = {
            "label": label,
            "valid_year": valid_year,
            "train_years": f"{train_years[0]}-{train_years[-1]}",
            "best_iteration": best_iter,
        }
        for metric in NDCG_METRICS:
            score = model.best_score["valid"][metric]
            metrics[metric.replace("@", "_at_")] = score
            print(f"    {metric.upper()}: {score:.6f}")
        fold_metrics.append(metrics)

    return fold_metrics, fold_curves, best_iters


def train_model(df: pd.DataFrame) -> tuple[lgb.Booster, dict, dict]:
    years = get_calendar_years(df)
    folds = build_walk_forward_folds(df)
    optuna_train_years = years[:-1]
    optuna_valid_year = years[-1]

    print("  Walk Forward 分割（評価用）:")
    for train_years, valid_year in folds:
        print(f"    学習 {train_years[0]}-{train_years[-1]} → 検証 {valid_year}")
    print(f"  Optuna 検証: 学習 {optuna_train_years[0]}-{optuna_train_years[-1]} → 検証 {optuna_valid_year}")
    print(f"  重み付け: 半減期 {RECENCY_HALF_LIFE_YEARS} 年")

    print(f"\n  Optuna 開始（{N_TRIALS} trials）...")
    study = run_optuna(subset_years(df, optuna_train_years), subset_years(df, [optuna_valid_year]))
    print(f"  Best NDCG@1: {study.best_value:.6f}")

    lgbm_params = build_lgbm_params(study.best_params, BASE_LGBM_RANKER_PARAMS)
    num_boost_round = study.best_params["num_boost_round"]

    fold_metrics, fold_curves, best_iters = run_walk_forward(df, folds, lgbm_params, num_boost_round)
    plot_learning_curves(fold_curves, LEARNING_CURVE_PNG)

    final_iter = max(int(np.median(best_iters)), MIN_BOOST_ROUND)
    print(f"\n  最終学習: {final_iter} rounds（WF best_iteration 中央値）")

    full_set, encoders = build_lgb_dataset(df, {}, fit_encoders=True)
    final_model = lgb.train(
        lgbm_params,
        full_set,
        num_boost_round=final_iter,
        callbacks=[lgb.log_evaluation(0)],
    )

    summary = {
        "model": "6艇ランキング_3連単予想",
        "pipeline": "LightGBM Ranker → Plackett–Luce",
        "train_years": list(TRAIN_YEARS),
        "n_races": len(df) // 6,
        "optuna_valid_year": optuna_valid_year,
        "optuna_best_ndcg_at_1": study.best_value,
        "walk_forward_folds": [
            {"train_years": f"{ty[0]}-{ty[-1]}", "valid_year": vy} for ty, vy in folds
        ],
        "walk_forward_fold_metrics": fold_metrics,
        "walk_forward_mean_ndcg_at_1": float(np.mean([m["ndcg_at_1"] for m in fold_metrics])),
        "walk_forward_mean_ndcg_at_3": float(np.mean([m["ndcg_at_3"] for m in fold_metrics])),
        "best_params": study.best_params,
        "n_trials": N_TRIALS,
        "num_boost_round": final_iter,
        "features": FEATURES,
        "top_n_list": TOP_N_LIST,
        "params": lgbm_params,
    }
    return final_model, encoders, summary


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    df = load_training_data()
    dates = pd.to_datetime(df["開催日"])
    print(f"  期間: {dates.min().date()} 〜 {dates.max().date()}")

    model, encoders, summary = train_model(df)

    pd.DataFrame(summary["walk_forward_fold_metrics"]).to_csv(
        WALK_FORWARD_CSV_PATH, index=False, encoding="UTF-8-sig",
    )
    print(f"\n  Walk Forward CSV: {WALK_FORWARD_CSV_PATH}")
    print(f"  平均 NDCG@1: {summary['walk_forward_mean_ndcg_at_1']:.6f}")
    print(f"  平均 NDCG@3: {summary['walk_forward_mean_ndcg_at_3']:.6f}")

    valid_year = summary["optuna_valid_year"]
    evaluate_model(
        model,
        subset_years(df, [valid_year]),
        encoders,
        f"検証{valid_year}",
        output_dir=OUTPUT_DIR,
    )

    print("\nSHAP分析")
    x, _, _, _, _ = prepare_dataset(df, encoders, fit_encoders=False)
    x_sample = x.sample(min(SHAP_SAMPLE_SIZE, len(x)), random_state=42)
    run_shap_analysis(
        model,
        x_sample,
        SHAP_BEESWARM_PNG,
        SHAP_WATERFALL_PNG,
        title="6艇ランキング_3連単予想",
    )
    save_shap_importance(model, x_sample, SHAP_IMPORTANCE_PNG, SHAP_IMPORTANCE_CSV)

    model.save_model(str(MODEL_PATH))
    joblib.dump(encoders, ENCODER_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n保存完了:")
    print(f"  {MODEL_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"  {CONFIG_PATH}")
    print(f"  {WALK_FORWARD_CSV_PATH}")
    print(f"  {LEARNING_CURVE_PNG}")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {PREDICTIONS_DETAIL_CSV_PATH}")
    print(f"  {SHAP_BEESWARM_PNG}")
    print(f"  {SHAP_WATERFALL_PNG}")
    print(f"  {SHAP_IMPORTANCE_PNG}")
    print(f"  {SHAP_IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()
