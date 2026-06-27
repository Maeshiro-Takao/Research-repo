"""
3連単オッズ予測モデル（LightGBM 回帰）

- データ: レース/選手/気象 CSV（2014〜2024年）
- 単位: 1レース1行（6艇の統計量 + 気象）
- 目的変数: log1p(的中3連単オッズ)
- 学習: Optuna（最終年ホールドアウト）+ Walk Forward 評価
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

from train_race_ranker import RACE_KEY, TRAIN_YEARS, load_race_data
from trifecta_training_utils import (
    RECENCY_HALF_LIFE_YEARS,
    UNKNOWN_CATEGORY_TOKEN,
    build_lgbm_params,
    compute_recency_weights,
    get_calendar_years,
    run_shap_analysis,
    setup_japanese_font,
)

BASE_DIR = Path(__file__).resolve().parent
TEST_YEARS = range(2025, 2026)

MODEL_DIR = BASE_DIR / "models" / "3連単オッズ予想"
OUTPUT_DIR = MODEL_DIR / "学習"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgbm_trifecta_odds.txt"
ENCODER_PATH = MODEL_DIR / "trifecta_odds_encoders.pkl"
CONFIG_PATH = MODEL_DIR / "trifecta_odds_config.json"
WALK_FORWARD_CSV_PATH = OUTPUT_DIR / "WalkForward評価.csv"
LEARNING_CURVE_PNG = OUTPUT_DIR / "学習曲線.png"
EVALUATION_CSV_PATH = OUTPUT_DIR / "評価結果.csv"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "予測結果.csv"
SHAP_BEESWARM_PNG = OUTPUT_DIR / "特徴量影響方向.png"
SHAP_WATERFALL_PNG = OUTPUT_DIR / "特徴量ウォーターフォール.png"
SHAP_IMPORTANCE_PNG = OUTPUT_DIR / "特徴量重要度.png"
SHAP_IMPORTANCE_CSV = OUTPUT_DIR / "特徴量重要度.csv"
TEST_OUTPUT_DIR = MODEL_DIR / "テスト"

REFERENCE_DATE = pd.Timestamp("2014-01-01")
CATEGORICAL_COLS = ["天気", "風向", "1号艇_級"]
AGG_NUMERIC_COLS = [
    "展示", "勝率", "2連率", "3連率", "当地勝率", "当地2連率", "当地3連率",
    "モーター勝率", "モーター2連率", "モーター3連率",
    "ボート勝率", "ボート2連率", "ボート3連率", "平均スタートタイミング",
    "1コース複勝率", "2コース複勝率", "3コース複勝率",
    "4コース複勝率", "5コース複勝率", "6コース複勝率",
]
BOAT1_NUMERIC_COLS = [
    "展示", "勝率", "2連率", "3連率", "当地勝率", "当地2連率", "当地3連率",
    "モーター勝率", "モーター2連率", "モーター3連率",
    "ボート勝率", "ボート2連率", "ボート3連率", "平均スタートタイミング",
    "1コース複勝率", "2コース複勝率", "3コース複勝率",
    "4コース複勝率", "5コース複勝率", "6コース複勝率", "体重",
]

NUMERIC_COLS = ["開催経過日", "風速", "波高"]
for col in AGG_NUMERIC_COLS:
    NUMERIC_COLS += [f"{col}_mean", f"{col}_max", f"{col}_min", f"{col}_std"]
for col in BOAT1_NUMERIC_COLS:
    NUMERIC_COLS.append(f"1号艇_{col}")
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

BASE_LGBM_REG_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": 42,
    "feature_pre_filter": False,
}

N_TRIALS = 25
SHAP_SAMPLE_SIZE = 2000
EARLY_STOPPING_ROUNDS = 50
MIN_BOOST_ROUND = 50
WF_MIN_TRAIN_YEARS = 7
REG_METRIC = "rmse"


def build_feature_names() -> list[str]:
    return FEATURES


def load_boat_data(years: range | list[int]) -> pd.DataFrame:
    df = load_race_data(years)
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    df = df.loc[sizes == 6].copy()
    df["__year"] = pd.to_datetime(df["開催日"]).dt.year
    return df


def subset_years(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    return df.loc[df["__year"].isin(years)].copy()


def _build_race_row(race_df: pd.DataFrame) -> dict | None:
    odds = pd.to_numeric(race_df["3連単オッズ"].iloc[0], errors="coerce")
    if pd.isna(odds) or odds <= 0:
        return None

    row: dict = {
        "開催日": race_df.iloc[0]["開催日"],
        "日目": race_df.iloc[0]["日目"],
        "レース": race_df.iloc[0]["レース"],
        "開催経過日": (pd.to_datetime(race_df.iloc[0]["開催日"]) - REFERENCE_DATE).days,
        "風速": pd.to_numeric(race_df["風速"].iloc[0], errors="coerce"),
        "波高": pd.to_numeric(race_df["波高"].iloc[0], errors="coerce"),
        "天気": race_df["天気"].iloc[0],
        "風向": race_df["風向"].iloc[0],
        "__target_log_odds": float(np.log1p(odds)),
        "__target_odds": float(odds),
    }

    for col in AGG_NUMERIC_COLS:
        vals = pd.to_numeric(race_df[col], errors="coerce").fillna(0.0)
        row[f"{col}_mean"] = float(vals.mean())
        row[f"{col}_max"] = float(vals.max())
        row[f"{col}_min"] = float(vals.min())
        row[f"{col}_std"] = float(vals.std(ddof=0))

    boat1 = race_df.loc[race_df["艇"] == 1]
    if len(boat1) == 1:
        b1 = boat1.iloc[0]
        row["1号艇_級"] = str(b1.get("級", ""))
        for col in BOAT1_NUMERIC_COLS:
            row[f"1号艇_{col}"] = float(pd.to_numeric(b1.get(col, 0), errors="coerce") or 0.0)
    else:
        row["1号艇_級"] = ""
        for col in BOAT1_NUMERIC_COLS:
            row[f"1号艇_{col}"] = 0.0

    return row


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


def prepare_odds_dataset(
    df: pd.DataFrame,
    encoders: dict | None = None,
    *,
    fit_encoders: bool = True,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict, np.ndarray, pd.DataFrame]:
    feature_names = feature_names or FEATURES
    encoders = encoders or {}
    rows: list[dict] = []

    for _, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        row = _build_race_row(race_df.sort_values("艇"))
        if row is not None:
            rows.append(row)

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise ValueError("オッズ付きレースがありません")

    x = meta.reindex(columns=feature_names, fill_value=0)
    for col in NUMERIC_COLS:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    x, encoders = _encode_categoricals(x, encoders, fit=fit_encoders)
    y = meta["__target_log_odds"].astype(float)
    weights = compute_recency_weights(meta["開催日"])
    return x, y, encoders, weights, meta


def build_lgb_dataset(df: pd.DataFrame, encoders: dict, *, fit_encoders: bool) -> tuple[lgb.Dataset, dict]:
    x, y, encoders, weights, _ = prepare_odds_dataset(df, encoders, fit_encoders=fit_encoders)
    return lgb.Dataset(x, label=y, weight=weights, free_raw_data=False), encoders


def predict_odds(model: lgb.Booster, x: pd.DataFrame) -> np.ndarray:
    return np.expm1(np.clip(model.predict(x), 0, None))


def load_model() -> tuple[lgb.Booster, dict, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {ENCODER_PATH}")
    features = FEATURES
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            features = json.load(f).get("features", FEATURES)
    return lgb.Booster(model_file=str(MODEL_PATH)), joblib.load(ENCODER_PATH), features


def evaluate_odds(
    model: lgb.Booster,
    df: pd.DataFrame,
    encoders: dict,
    label: str,
    *,
    feature_names: list[str] | None = None,
    output_metrics_path: Path | None = None,
    output_predictions_path: Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    feature_names = feature_names or FEATURES
    x, y_log, _, _, meta = prepare_odds_dataset(
        df, encoders, fit_encoders=False, feature_names=feature_names,
    )
    pred_log = model.predict(x)
    actual_odds = meta["__target_odds"].values
    pred_odds = np.expm1(np.clip(pred_log, 0, None))

    log_err = pred_log - y_log.values
    odds_err = pred_odds - actual_odds
    rel_err = np.abs(odds_err) / np.maximum(actual_odds, 1e-6)

    metrics = {
        "label": label,
        "n_races": len(meta),
        "rmse_log": float(np.sqrt(np.mean(log_err ** 2))),
        "mae_log": float(np.mean(np.abs(log_err))),
        "rmse_odds": float(np.sqrt(np.mean(odds_err ** 2))),
        "mae_odds": float(np.mean(np.abs(odds_err))),
        "spearman": float(meta["__target_odds"].corr(pd.Series(pred_odds), method="spearman")),
        "within_20pct_rate": float(np.mean(rel_err <= 0.2)),
        "within_50pct_rate": float(np.mean(rel_err <= 0.5)),
    }

    print(f"\n=== 3連単オッズ評価 ({label}) ===")
    print(f"  レース数: {metrics['n_races']}")
    print(f"  RMSE (log): {metrics['rmse_log']:.4f}")
    print(f"  MAE (log):  {metrics['mae_log']:.4f}")
    print(f"  RMSE (オッズ): {metrics['rmse_odds']:.2f}")
    print(f"  MAE (オッズ):  {metrics['mae_odds']:.2f}")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    print(f"  ±20%以内: {metrics['within_20pct_rate']:.2%}")
    print(f"  ±50%以内: {metrics['within_50pct_rate']:.2%}")

    pred_df = pd.DataFrame({
        "開催日": meta["開催日"],
        "日目": meta["日目"],
        "レース": meta["レース"],
        "実際オッズ": actual_odds,
        "予測オッズ": pred_odds,
        "誤差": odds_err,
        "相対誤差": rel_err,
    })

    if output_metrics_path is not None:
        pd.DataFrame([metrics]).to_csv(output_metrics_path, index=False, encoding="UTF-8-sig")
    if output_predictions_path is not None:
        pred_df.to_csv(output_predictions_path, index=False, encoding="UTF-8-sig")

    return metrics, pred_df


def save_shap_importance(model: lgb.Booster, x_sample: pd.DataFrame, png_path: Path, csv_path: Path) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    feature_names = [str(c) for c in x_sample.columns]
    importance = pd.DataFrame({
        "特徴量": feature_names,
        "平均絶対SHAP": np.abs(shap_values).mean(axis=0),
    }).sort_values("平均絶対SHAP", ascending=False)
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


def build_walk_forward_folds(df: pd.DataFrame) -> list[tuple[list[int], int]]:
    years = get_calendar_years(df)
    folds = [(years[:idx], years[idx]) for idx in range(WF_MIN_TRAIN_YEARS, len(years))]
    if not folds:
        raise ValueError("Walk Forward 分割を構築できません")
    return folds


def suggest_lgbm_params(trial: optuna.Trial) -> tuple[dict, int]:
    params = {
        **BASE_LGBM_REG_PARAMS,
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


def run_optuna(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> optuna.Study:
    train_set, encoders = build_lgb_dataset(train_df, {}, fit_encoders=True)
    valid_set, _ = build_lgb_dataset(valid_df, encoders, fit_encoders=False)

    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = suggest_lgbm_params(trial)
        model = lgb.train(
            params, train_set, num_boost_round=num_boost_round,
            valid_sets=[valid_set], valid_names=["valid"],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return float(model.best_score["valid"][REG_METRIC])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    return study


def plot_learning_curves(fold_curves: list[tuple[str, dict]], png_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, evals in fold_curves:
        curve = evals.get("valid", {}).get(REG_METRIC, [])
        if curve:
            axes[0].plot(range(1, len(curve) + 1), curve, label=label)
    axes[0].set_title("Walk Forward 検証 RMSE")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    last_label, last_evals = fold_curves[-1]
    for split, jp in (("train", "学習"), ("valid", "検証")):
        curve = last_evals.get(split, {}).get(REG_METRIC, [])
        if curve:
            axes[1].plot(range(1, len(curve) + 1), curve, label=jp)
    axes[1].set_title(f"最終フォールド {last_label}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
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
            lgbm_params, train_set, num_boost_round=num_boost_round,
            valid_sets=[train_set, valid_set], valid_names=["train", "valid"],
            callbacks=[
                lgb.record_evaluation(evals),
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_iter = model.best_iteration or num_boost_round
        best_iters.append(best_iter)
        fold_curves.append((label, evals))

        wf_metrics, _ = evaluate_odds(model, subset_years(df, [valid_year]), encoders, label)
        fold_metrics.append({
            "label": label,
            "valid_year": valid_year,
            "train_years": f"{train_years[0]}-{train_years[-1]}",
            "best_iteration": best_iter,
            "valid_rmse_log": model.best_score["valid"][REG_METRIC],
            **wf_metrics,
        })

    return fold_metrics, fold_curves, best_iters


def train_model(df: pd.DataFrame) -> tuple[lgb.Booster, dict, dict]:
    years = get_calendar_years(df)
    folds = build_walk_forward_folds(df)
    optuna_train_years, optuna_valid_year = years[:-1], years[-1]

    print("  Walk Forward 分割:")
    for train_years, valid_year in folds:
        print(f"    学習 {train_years[0]}-{train_years[-1]} → 検証 {valid_year}")
    print(f"  Optuna 検証: 学習 {optuna_train_years[0]}-{optuna_train_years[-1]} → 検証 {optuna_valid_year}")

    print(f"\n  Optuna 開始（{N_TRIALS} trials）...")
    study = run_optuna(subset_years(df, optuna_train_years), subset_years(df, [optuna_valid_year]))
    print(f"  Best RMSE (log): {study.best_value:.6f}")

    lgbm_params = build_lgbm_params(study.best_params, BASE_LGBM_REG_PARAMS)
    num_boost_round = study.best_params["num_boost_round"]

    fold_metrics, fold_curves, best_iters = run_walk_forward(df, folds, lgbm_params, num_boost_round)
    plot_learning_curves(fold_curves, LEARNING_CURVE_PNG)

    final_iter = max(int(np.median(best_iters)), MIN_BOOST_ROUND)
    full_set, encoders = build_lgb_dataset(df, {}, fit_encoders=True)
    final_model = lgb.train(
        lgbm_params, full_set, num_boost_round=final_iter,
        callbacks=[lgb.log_evaluation(0)],
    )

    summary = {
        "model": "3連単オッズ予想",
        "target": "log1p(3連単オッズ)",
        "train_years": list(TRAIN_YEARS),
        "n_races": len(df.groupby(RACE_KEY)),
        "optuna_valid_year": optuna_valid_year,
        "optuna_best_rmse_log": study.best_value,
        "walk_forward_fold_metrics": fold_metrics,
        "walk_forward_mean_mae_odds": float(np.mean([m["mae_odds"] for m in fold_metrics])),
        "best_params": study.best_params,
        "n_trials": N_TRIALS,
        "num_boost_round": final_iter,
        "features": FEATURES,
        "params": lgbm_params,
    }
    return final_model, encoders, summary


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    print("データ読み込み: 2014〜2024年")
    df = load_boat_data(TRAIN_YEARS)
    print(f"  レース数: {len(df) // 6} / 特徴量数: {len(FEATURES)}")

    model, encoders, summary = train_model(df)

    pd.DataFrame(summary["walk_forward_fold_metrics"]).to_csv(
        WALK_FORWARD_CSV_PATH, index=False, encoding="UTF-8-sig",
    )

    valid_year = summary["optuna_valid_year"]
    evaluate_odds(
        model,
        subset_years(df, [valid_year]),
        encoders,
        f"検証{valid_year}",
        output_metrics_path=EVALUATION_CSV_PATH,
        output_predictions_path=PREDICTIONS_CSV_PATH,
    )

    print("\nSHAP分析")
    x, _, _, _, _ = prepare_odds_dataset(df, encoders, fit_encoders=False)
    x_sample = x.sample(min(SHAP_SAMPLE_SIZE, len(x)), random_state=42)
    run_shap_analysis(model, x_sample, SHAP_BEESWARM_PNG, SHAP_WATERFALL_PNG, title="3連単オッズ予想")
    save_shap_importance(model, x_sample, SHAP_IMPORTANCE_PNG, SHAP_IMPORTANCE_CSV)

    model.save_model(str(MODEL_PATH))
    joblib.dump(encoders, ENCODER_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n保存完了:")
    print(f"  {MODEL_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"  {CONFIG_PATH}")


if __name__ == "__main__":
    main()
