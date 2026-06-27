"""
3連単オッズ予測・二段階モデル

Step1: オッズ帯分類（LightGBM Multiclass）
Step2: クラス別 log(オッズ) 回帰（LightGBM Regressor）
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import train_trifecta_odds as baseline
from train_race_ranker import RACE_KEY
from train_trifecta_odds import (
    EARLY_STOPPING_ROUNDS,
    FEATURES,
    MIN_BOOST_ROUND,
    N_TRIALS,
    REG_METRIC,
    SHAP_SAMPLE_SIZE,
    TRAIN_YEARS,
    WF_MIN_TRAIN_YEARS,
    build_lgb_dataset,
    load_boat_data,
    prepare_odds_dataset,
    subset_years,
)
from trifecta_training_utils import (
    build_lgbm_params,
    get_calendar_years,
    run_shap_analysis,
    setup_japanese_font,
)

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
TEST_YEARS = range(2025, 2026)

MODEL_DIR = BASE_DIR / "models" / "3連単オッズ予想" / "二段階"
OUTPUT_DIR = MODEL_DIR / "学習"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR = MODEL_DIR / "テスト"

CLASSIFIER_PATH = MODEL_DIR / "lgbm_odds_classifier.txt"
REGRESSOR_DIR = MODEL_DIR / "クラス別回帰"
REGRESSOR_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_REGRESSOR_PATH = MODEL_DIR / "lgbm_odds_global_regressor.txt"
ENCODER_PATH = MODEL_DIR / "trifecta_odds_two_stage_encoders.pkl"
CONFIG_PATH = MODEL_DIR / "trifecta_odds_two_stage_config.json"

WALK_FORWARD_CSV_PATH = OUTPUT_DIR / "WalkForward評価.csv"
CLASSIFIER_WF_CSV_PATH = OUTPUT_DIR / "分類_WalkForward評価.csv"
CONFUSION_MATRIX_CSV_PATH = OUTPUT_DIR / "混同行列.csv"
CONFUSION_MATRIX_PNG = OUTPUT_DIR / "混同行列.png"
LEARNING_CURVE_CLF_PNG = OUTPUT_DIR / "分類_学習曲線.png"
LEARNING_CURVE_REG_PNG = OUTPUT_DIR / "回帰_学習曲線.png"
EVALUATION_CSV_PATH = OUTPUT_DIR / "評価結果.csv"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "予測結果.csv"
SHAP_CLF_BEESWARM_PNG = OUTPUT_DIR / "分類_特徴量影響方向.png"
SHAP_CLF_WATERFALL_PNG = OUTPUT_DIR / "分類_特徴量ウォーターフォール.png"
SHAP_CLF_IMPORTANCE_PNG = OUTPUT_DIR / "分類_特徴量重要度.png"
SHAP_CLF_IMPORTANCE_CSV = OUTPUT_DIR / "分類_特徴量重要度.csv"
SHAP_REG_BEESWARM_PNG = OUTPUT_DIR / "回帰_特徴量影響方向.png"
SHAP_REG_WATERFALL_PNG = OUTPUT_DIR / "回帰_特徴量ウォーターフォール.png"
SHAP_REG_IMPORTANCE_PNG = OUTPUT_DIR / "回帰_特徴量重要度.png"
SHAP_REG_IMPORTANCE_CSV = OUTPUT_DIR / "回帰_特徴量重要度.csv"

NUM_CLASSES = 5
MIN_CLASS_SAMPLES = 50
ODDS_BAND_LABELS = ["0-10倍", "10-30倍", "30-100倍", "100-300倍", "300倍以上"]

BASE_LGBM_CLF_PARAMS = {
    "objective": "multiclass",
    "num_class": NUM_CLASSES,
    "metric": "multi_logloss",
    "verbosity": -1,
    "seed": RANDOM_SEED,
    "feature_pre_filter": False,
}

EVAL_METRICS = [
    "rmse_log", "mae_log", "rmse_odds", "mae_odds",
    "spearman", "within_20pct_rate", "within_50pct_rate",
]


def odds_to_band(odds: float) -> int:
    if odds < 10:
        return 0
    if odds < 30:
        return 1
    if odds < 100:
        return 2
    if odds < 300:
        return 3
    return 4


def band_label(band: int) -> str:
    return ODDS_BAND_LABELS[band]


def regressor_path(band: int) -> Path:
    return REGRESSOR_DIR / f"lgbm_odds_regressor_band{band}.txt"


def prepare_with_band(
    df: pd.DataFrame,
    encoders: dict | None = None,
    *,
    fit_encoders: bool = True,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict, np.ndarray, pd.DataFrame]:
    x, y_log, encoders, weights, meta = prepare_odds_dataset(
        df, encoders, fit_encoders=fit_encoders, feature_names=feature_names,
    )
    y_band = meta["__target_odds"].apply(odds_to_band).astype(int)
    return x, y_log, y_band, encoders, weights, meta


def build_classifier_dataset(
    df: pd.DataFrame, encoders: dict, *, fit_encoders: bool,
) -> tuple[lgb.Dataset, dict]:
    x, _, y_band, encoders, weights, _ = prepare_with_band(df, encoders, fit_encoders=fit_encoders)
    return lgb.Dataset(x, label=y_band, weight=weights, free_raw_data=False), encoders


def build_classifier_lgbm_params(best_params: dict) -> dict:
    params = {**BASE_LGBM_CLF_PARAMS}
    mapping = {
        "clf_num_leaves": "num_leaves",
        "clf_max_depth": "max_depth",
        "clf_learning_rate": "learning_rate",
        "clf_min_data_in_leaf": "min_data_in_leaf",
        "clf_feature_fraction": "feature_fraction",
        "clf_bagging_fraction": "bagging_fraction",
        "clf_bagging_freq": "bagging_freq",
        "clf_lambda_l1": "lambda_l1",
        "clf_lambda_l2": "lambda_l2",
    }
    for src, dst in mapping.items():
        params[dst] = best_params[src]
    return params


def build_regressor_lgbm_params(best_params: dict) -> dict:
    params = build_lgbm_params(best_params, baseline.BASE_LGBM_REG_PARAMS)
    params["seed"] = RANDOM_SEED
    return params


def suggest_classifier_params(trial: optuna.Trial) -> tuple[dict, int]:
    params = {
        **BASE_LGBM_CLF_PARAMS,
        "num_leaves": trial.suggest_int("clf_num_leaves", 16, 96),
        "max_depth": trial.suggest_int("clf_max_depth", 4, 10),
        "learning_rate": trial.suggest_float("clf_learning_rate", 0.02, 0.15, log=True),
        "min_data_in_leaf": trial.suggest_int("clf_min_data_in_leaf", 30, 150),
        "feature_fraction": trial.suggest_float("clf_feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("clf_bagging_fraction", 0.7, 1.0),
        "bagging_freq": trial.suggest_int("clf_bagging_freq", 1, 5),
        "lambda_l1": trial.suggest_float("clf_lambda_l1", 1e-3, 5.0, log=True),
        "lambda_l2": trial.suggest_float("clf_lambda_l2", 1e-3, 5.0, log=True),
    }
    rounds = trial.suggest_int("clf_num_boost_round", 100, 400)
    return params, rounds


def suggest_regressor_params(trial: optuna.Trial) -> tuple[dict, int]:
    params, rounds = baseline.suggest_lgbm_params(trial)
    params["seed"] = RANDOM_SEED
    return params, rounds


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    mf1 = macro_f1(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print(f"\n=== オッズ帯分類 ({label}) ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {mf1:.4f}")
    return {
        "label": label,
        "accuracy": acc,
        "macro_f1": mf1,
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm: np.ndarray, png_path: Path, csv_path: Path, title: str) -> None:
    cm_df = pd.DataFrame(
        cm,
        index=[f"実際:{band_label(i)}" for i in range(NUM_CLASSES)],
        columns=[f"予測:{band_label(i)}" for i in range(NUM_CLASSES)],
    )
    cm_df.to_csv(csv_path, encoding="UTF-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ODDS_BAND_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(ODDS_BAND_LABELS)
    ax.set_xlabel("予測")
    ax.set_ylabel("実際")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  混同行列: {png_path}")


def train_class_regressors(
    train_df: pd.DataFrame,
    encoders: dict,
    reg_params: dict,
    num_boost_round: int,
    global_regressor: lgb.Booster | None = None,
) -> tuple[dict[int, lgb.Booster], lgb.Booster]:
    x, y_log, y_band, encoders, weights, meta = prepare_with_band(
        train_df, encoders, fit_encoders=False,
    )
    regressors: dict[int, lgb.Booster] = {}

    if global_regressor is None:
        full_set, _ = build_lgb_dataset(train_df, encoders, fit_encoders=False)
        global_regressor = lgb.train(
            reg_params, full_set, num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(0)],
        )

    for band in range(NUM_CLASSES):
        mask = y_band.values == band
        if mask.sum() < MIN_CLASS_SAMPLES:
            regressors[band] = global_regressor
            continue
        band_set = lgb.Dataset(
            x.loc[mask], label=y_log.loc[mask], weight=weights[mask], free_raw_data=False,
        )
        regressors[band] = lgb.train(
            reg_params, band_set, num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(0)],
        )
    return regressors, global_regressor


def predict_two_stage_log(
    classifier: lgb.Booster,
    regressors: dict[int, lgb.Booster],
    global_regressor: lgb.Booster,
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    pred_band = np.argmax(classifier.predict(x), axis=1)
    pred_log = np.zeros(len(x), dtype=float)
    for band in range(NUM_CLASSES):
        mask = pred_band == band
        if not mask.any():
            continue
        reg = regressors.get(band, global_regressor)
        pred_log[mask] = reg.predict(x.loc[mask])
    return pred_log, pred_band


def evaluate_two_stage(
    classifier: lgb.Booster,
    regressors: dict[int, lgb.Booster],
    global_regressor: lgb.Booster,
    df: pd.DataFrame,
    encoders: dict,
    label: str,
    *,
    feature_names: list[str] | None = None,
) -> tuple[dict, dict, pd.DataFrame]:
    feature_names = feature_names or FEATURES
    x, y_log, y_band, _, _, meta = prepare_with_band(
        df, encoders, fit_encoders=False, feature_names=feature_names,
    )
    pred_log, pred_band = predict_two_stage_log(classifier, regressors, global_regressor, x)
    clf_metrics = evaluate_classification(y_band.values, pred_band, label)

    actual_odds = meta["__target_odds"].values
    pred_odds = np.expm1(np.clip(pred_log, 0, None))
    log_err = pred_log - y_log.values
    odds_err = pred_odds - actual_odds
    rel_err = np.abs(odds_err) / np.maximum(actual_odds, 1e-6)

    reg_metrics = {
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
    for key in EVAL_METRICS:
        val = reg_metrics[key]
        if "rate" in key:
            print(f"  {key}: {val:.2%}")
        elif key in ("rmse_odds", "mae_odds"):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val:.4f}")

    pred_df = pd.DataFrame({
        "開催日": meta["開催日"], "日目": meta["日目"], "レース": meta["レース"],
        "実際オッズ": actual_odds, "予測オッズ": pred_odds,
        "実際帯": [band_label(b) for b in y_band],
        "予測帯": [band_label(b) for b in pred_band],
        "誤差": odds_err, "相対誤差": rel_err,
    })
    return reg_metrics, clf_metrics, pred_df


def metrics_row(metrics: dict, *, split: str) -> dict:
    row = {"split": split, "label": metrics.get("label", split)}
    for key in EVAL_METRICS:
        row[key] = metrics.get(key)
    if "accuracy" in metrics:
        row["accuracy"] = metrics["accuracy"]
        row["macro_f1"] = metrics["macro_f1"]
    return row


def run_classifier_optuna(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> optuna.Study:
    train_set, encoders = build_classifier_dataset(train_df, {}, fit_encoders=True)
    valid_set, _ = build_classifier_dataset(valid_df, encoders, fit_encoders=False)
    x_valid, _, y_valid, _, _, _ = prepare_with_band(valid_df, encoders, fit_encoders=False)

    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = suggest_classifier_params(trial)
        model = lgb.train(
            params, train_set, num_boost_round=num_boost_round,
            valid_sets=[valid_set], valid_names=["valid"],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        pred_band = np.argmax(model.predict(x_valid), axis=1)
        return macro_f1(y_valid.values, pred_band)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    return study


def run_regressor_optuna(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)

    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = suggest_regressor_params(trial)
        train_set, encoders = build_lgb_dataset(train_df, {}, fit_encoders=True)
        valid_set, _ = build_lgb_dataset(valid_df, encoders, fit_encoders=False)
        model = lgb.train(
            params, train_set, num_boost_round=num_boost_round,
            valid_sets=[valid_set], valid_names=["valid"],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return float(model.best_score["valid"][REG_METRIC])

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    return study


def plot_clf_learning_curves(fold_curves: list[tuple[str, dict]], png_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric = "multi_logloss"
    for label, evals in fold_curves:
        curve = evals.get("valid", {}).get(metric, [])
        if curve:
            axes[0].plot(range(1, len(curve) + 1), curve, label=label)
    axes[0].set_title("Walk Forward 検証 multi_logloss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    last_label, last_evals = fold_curves[-1]
    for split, jp in (("train", "学習"), ("valid", "検証")):
        curve = last_evals.get(split, {}).get(metric, [])
        if curve:
            axes[1].plot(range(1, len(curve) + 1), curve, label=jp)
    axes[1].set_title(f"最終フォールド {last_label}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_shap_importance(model: lgb.Booster, x_sample: pd.DataFrame, png_path: Path, csv_path: Path) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = np.abs(np.array(shap_values)).mean(axis=0)
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
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def train_fold_models(
    train_df: pd.DataFrame,
    clf_params: dict,
    clf_rounds: int,
    reg_params: dict,
    reg_rounds: int,
) -> tuple[lgb.Booster, dict[int, lgb.Booster], lgb.Booster, dict]:
    train_clf_set, encoders = build_classifier_dataset(train_df, {}, fit_encoders=True)
    classifier = lgb.train(
        clf_params, train_clf_set, num_boost_round=clf_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    global_set, _ = build_lgb_dataset(train_df, encoders, fit_encoders=False)
    global_regressor = lgb.train(
        reg_params, global_set, num_boost_round=reg_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    regressors, global_regressor = train_class_regressors(
        train_df, encoders, reg_params, reg_rounds, global_regressor=global_regressor,
    )
    return classifier, regressors, global_regressor, encoders


def plot_reg_learning_curves(fold_curves: list[tuple[str, dict]], png_path: Path) -> None:
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
    clf_params: dict,
    clf_rounds: int,
    reg_params: dict,
    reg_rounds: int,
) -> tuple[list[dict], list[dict], list[tuple[str, dict]], list[tuple[str, dict]]]:
    wf_rows: list[dict] = []
    clf_rows: list[dict] = []
    clf_curves: list[tuple[str, dict]] = []
    reg_curves: list[tuple[str, dict]] = []

    print("\nWalk Forward 評価（OOS）")
    for train_years, valid_year in folds:
        label = f"学習{train_years[0]}-{train_years[-1]}→検証{valid_year}"
        train_df = subset_years(df, train_years)
        valid_df = subset_years(df, [valid_year])

        train_clf_set, encoders = build_classifier_dataset(train_df, {}, fit_encoders=True)
        valid_clf_set, _ = build_classifier_dataset(valid_df, encoders, fit_encoders=False)
        clf_evals: dict = {}
        classifier = lgb.train(
            clf_params, train_clf_set, num_boost_round=clf_rounds,
            valid_sets=[train_clf_set, valid_clf_set], valid_names=["train", "valid"],
            callbacks=[
                lgb.record_evaluation(clf_evals),
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        clf_curves.append((label, clf_evals))

        train_reg_set, _ = build_lgb_dataset(train_df, encoders, fit_encoders=False)
        valid_reg_set, _ = build_lgb_dataset(valid_df, encoders, fit_encoders=False)
        reg_evals: dict = {}
        global_regressor = lgb.train(
            reg_params, train_reg_set, num_boost_round=reg_rounds,
            valid_sets=[train_reg_set, valid_reg_set], valid_names=["train", "valid"],
            callbacks=[
                lgb.record_evaluation(reg_evals),
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        reg_curves.append((label, reg_evals))
        regressors, global_regressor = train_class_regressors(
            train_df, encoders, reg_params, reg_rounds, global_regressor=global_regressor,
        )

        ts_reg, ts_clf, _ = evaluate_two_stage(
            classifier, regressors, global_regressor, valid_df, encoders, label,
        )
        wf_rows.append(metrics_row(ts_reg, split="walk_forward") | {
            "valid_year": valid_year, "train_years": f"{train_years[0]}-{train_years[-1]}",
        })
        clf_rows.append({
            "label": label, "valid_year": valid_year,
            "accuracy": ts_clf["accuracy"], "macro_f1": ts_clf["macro_f1"],
            "confusion_matrix": ts_clf["confusion_matrix"],
        })

    return wf_rows, clf_rows, clf_curves, reg_curves


def save_models(
    classifier: lgb.Booster,
    regressors: dict[int, lgb.Booster],
    global_regressor: lgb.Booster,
    encoders: dict,
) -> None:
    classifier.save_model(str(CLASSIFIER_PATH))
    global_regressor.save_model(str(GLOBAL_REGRESSOR_PATH))
    for band, model in regressors.items():
        model.save_model(str(regressor_path(band)))
    joblib.dump(encoders, ENCODER_PATH)


def load_two_stage_model() -> tuple[lgb.Booster, dict[int, lgb.Booster], lgb.Booster, dict, list[str]]:
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"分類器が見つかりません: {CLASSIFIER_PATH}")
    classifier = lgb.Booster(model_file=str(CLASSIFIER_PATH))
    global_regressor = lgb.Booster(model_file=str(GLOBAL_REGRESSOR_PATH))
    regressors: dict[int, lgb.Booster] = {}
    for band in range(NUM_CLASSES):
        path = regressor_path(band)
        regressors[band] = lgb.Booster(model_file=str(path)) if path.exists() else global_regressor
    encoders = joblib.load(ENCODER_PATH)
    features = FEATURES
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            features = json.load(f).get("features", FEATURES)
    return classifier, regressors, global_regressor, encoders, features


def train_model(df: pd.DataFrame) -> dict:
    np.random.seed(RANDOM_SEED)
    years = get_calendar_years(df)
    folds = [(years[:i], years[i]) for i in range(WF_MIN_TRAIN_YEARS, len(years))]
    optuna_train, optuna_valid = years[:-1], years[-1]

    print("  Walk Forward 分割:")
    for ty, vy in folds:
        print(f"    学習 {ty[0]}-{ty[-1]} → 検証 {vy}")

    print(f"\n  Optuna 分類器（{N_TRIALS} trials）...")
    clf_study = run_classifier_optuna(subset_years(df, optuna_train), subset_years(df, [optuna_valid]))
    print(f"  Best Macro F1: {clf_study.best_value:.4f}")

    print(f"\n  Optuna 回帰（{N_TRIALS} trials）...")
    reg_study = run_regressor_optuna(subset_years(df, optuna_train), subset_years(df, [optuna_valid]))
    print(f"  Best RMSE (log): {reg_study.best_value:.6f}")

    clf_params = build_classifier_lgbm_params(clf_study.best_params)
    reg_params = build_regressor_lgbm_params(reg_study.best_params)
    clf_rounds = clf_study.best_params["clf_num_boost_round"]
    reg_rounds = reg_study.best_params["num_boost_round"]

    wf_rows, clf_rows, clf_curves, reg_curves = run_walk_forward(
        df, folds, clf_params, clf_rounds, reg_params, reg_rounds,
    )
    plot_clf_learning_curves(clf_curves, LEARNING_CURVE_CLF_PNG)
    plot_reg_learning_curves(reg_curves, LEARNING_CURVE_REG_PNG)

    classifier, regressors, global_regressor, encoders = train_fold_models(
        df, clf_params, clf_rounds, reg_params, reg_rounds,
    )
    save_models(classifier, regressors, global_regressor, encoders)

    valid_df = subset_years(df, [optuna_valid])
    ts_reg, ts_clf, pred_df = evaluate_two_stage(
        classifier, regressors, global_regressor, valid_df, encoders, f"検証{optuna_valid}",
    )
    save_confusion_matrix(
        np.array(ts_clf["confusion_matrix"]), CONFUSION_MATRIX_PNG, CONFUSION_MATRIX_CSV_PATH,
        f"混同行列（検証{optuna_valid}）",
    )
    pd.DataFrame([ts_reg]).to_csv(EVALUATION_CSV_PATH, index=False, encoding="UTF-8-sig")
    pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="UTF-8-sig")

    return {
        "model": "3連単オッズ予想_二段階",
        "random_seed": RANDOM_SEED,
        "odds_bands": ODDS_BAND_LABELS,
        "optuna_valid_year": optuna_valid,
        "classifier_best_params": clf_study.best_params,
        "regressor_best_params": reg_study.best_params,
        "walk_forward_classification": clf_rows,
        "walk_forward_regression": wf_rows,
        "features": FEATURES,
        "holdout_regression": metrics_row(ts_reg, split=f"検証{optuna_valid}"),
    }


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(RANDOM_SEED)
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    print("データ読み込み: 2014〜2024年")
    df = load_boat_data(TRAIN_YEARS)
    print(f"  レース数: {len(df) // 6} / 特徴量数: {len(FEATURES)}")
    band_counts = df.groupby(RACE_KEY)["3連単オッズ"].first().apply(
        lambda v: band_label(odds_to_band(float(v))) if pd.notna(v) else None
    ).value_counts()
    print("  オッズ帯分布:")
    for name, cnt in band_counts.items():
        print(f"    {name}: {cnt}")

    summary = train_model(df)
    clf_rows = summary["walk_forward_classification"]
    classifier, regressors, global_regressor, encoders, _ = load_two_stage_model()

    pd.DataFrame(clf_rows).drop(columns=["confusion_matrix"], errors="ignore").to_csv(
        CLASSIFIER_WF_CSV_PATH, index=False, encoding="UTF-8-sig",
    )
    pd.DataFrame(summary["walk_forward_regression"]).to_csv(
        WALK_FORWARD_CSV_PATH, index=False, encoding="UTF-8-sig",
    )

    wf = pd.DataFrame(summary["walk_forward_regression"])
    if not wf.empty:
        print("\n=== Walk Forward 平均 ===")
        for key in EVAL_METRICS:
            print(f"  平均 {key}: {wf[key].mean():.4f}")

    print("\nSHAP分析（分類器）")
    x, _, _, _, _, _ = prepare_with_band(df, encoders, fit_encoders=False)
    x_sample = x.sample(min(SHAP_SAMPLE_SIZE, len(x)), random_state=RANDOM_SEED)
    run_shap_analysis(classifier, x_sample, SHAP_CLF_BEESWARM_PNG, SHAP_CLF_WATERFALL_PNG, title="オッズ帯分類")
    save_shap_importance(classifier, x_sample, SHAP_CLF_IMPORTANCE_PNG, SHAP_CLF_IMPORTANCE_CSV)

    print("\nSHAP分析（全体回帰）")
    run_shap_analysis(global_regressor, x_sample, SHAP_REG_BEESWARM_PNG, SHAP_REG_WATERFALL_PNG, title="クラス別回帰(全体)")
    save_shap_importance(global_regressor, x_sample, SHAP_REG_IMPORTANCE_PNG, SHAP_REG_IMPORTANCE_CSV)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n保存完了:")
    print(f"  {CLASSIFIER_PATH}")
    print(f"  {GLOBAL_REGRESSOR_PATH}")
    print(f"  {WALK_FORWARD_CSV_PATH}")
    print(f"  {EVALUATION_CSV_PATH}")


if __name__ == "__main__":
    main()
