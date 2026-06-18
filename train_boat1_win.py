"""
1号艇勝利予測モデル（LightGBM Classifier）

P(1号艇が1着になる) を予測する二値分類モデル。

- 学習: 2014〜2023年
- 検証: 2024年
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from trifecta_training_utils import (
    CATEGORICAL_FEATURES,
    build_feature_list,
    run_shap_analysis,
    setup_japanese_font,
)

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "レースデータ" / "丸亀学習用_レースデータ.csv"
MODEL_DIR = BASE_DIR / "models" / "1号艇勝利予測"
TRAIN_OUTPUT_DIR = MODEL_DIR / "学習"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lgbm_boat1_win_model.txt"
CLASSIFIER_PATH = MODEL_DIR / "boat1_win_classifier.pkl"
ENCODER_PATH = MODEL_DIR / "boat1_win_encoders.pkl"
METRICS_PATH = TRAIN_OUTPUT_DIR / "評価結果.json"
OPTUNA_SUMMARY_PATH = TRAIN_OUTPUT_DIR / "最適化結果.json"
CONFUSION_MATRIX_CSV_PATH = TRAIN_OUTPUT_DIR / "混同行列.csv"
CONFUSION_MATRIX_PNG_PATH = TRAIN_OUTPUT_DIR / "混同行列.png"
PREDICTIONS_CSV_PATH = TRAIN_OUTPUT_DIR / "検証_予測結果.csv"
SHAP_BEESWARM_PNG_PATH = TRAIN_OUTPUT_DIR / "特徴量影響方向.png"
SHAP_WATERFALL_PNG_PATH = TRAIN_OUTPUT_DIR / "特徴量ウォーターフォール.png"

RACE_KEY = ["開催日", "日目", "レース"]
TARGET_BOAT = 1
TRAIN_YEAR_END = 2023
VALID_YEAR = 2024
SHAP_SAMPLE_SIZE = 2000
N_TRIALS = 50

FEATURES: list[str] = []

FIXED_CLASSIFIER_PARAMS = {
    "objective": "binary",
    "random_state": 42,
    "verbosity": -1,
}


def configure_features(df: pd.DataFrame) -> None:
    global FEATURES
    FEATURES = build_feature_list(df)


def load_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}")
    df = pd.read_csv(RACE_DATA_PATH)
    configure_features(df)
    df = filter_complete_races(df)
    n_races = df.drop_duplicates(RACE_KEY).shape[0]
    boat1_wins = count_boat1_wins(df)
    print(f"  対象: {len(df)} 行 / {n_races} レース")
    print(f"  1着=1号艇: {boat1_wins} レース ({boat1_wins / n_races:.1%})")
    print(f"  特徴量数: {len(FEATURES)}")
    return df


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    return df.loc[sizes == 6].copy()


def count_boat1_wins(df: pd.DataFrame) -> int:
    n = 0
    for _, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        if int(race_df.loc[race_df["着"] == 1, "艇"].iloc[0]) == TARGET_BOAT:
            n += 1
    return n


def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = pd.to_datetime(df["開催日"]).dt.year
    train_df = df.loc[years <= TRAIN_YEAR_END].copy()
    valid_df = df.loc[years == VALID_YEAR].copy()
    return train_df, valid_df


def prepare_race_samples(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit_encoders: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict, pd.DataFrame]:
    """
    レース単位の1号艇行を抽出し、特徴量と目的変数を作成する。

    目的変数: 1着=1号艇 → 1、それ以外 → 0
    特徴量作成はランキングモデルと同様（展示順位・展示差をレース内で算出）
    """
    df = filter_complete_races(df)
    df = df.sort_values(RACE_KEY).reset_index(drop=True)
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")

    boat1 = df.loc[df["艇"].astype(int) == TARGET_BOAT].copy()
    meta = boat1[RACE_KEY + ["開催日", "着"]].copy()
    boat1 = boat1.drop(columns=["日目", "選手名"], errors="ignore")

    encoders = encoders or {}
    for col in CATEGORICAL_FEATURES:
        if fit_encoders and col not in encoders:
            encoders[col] = {v: i for i, v in enumerate(boat1[col].astype(str).unique())}
        boat1[col] = boat1[col].astype(str).map(encoders[col]).fillna(-1).astype(int)

    y = (boat1["着"].astype(int) == 1).astype(int)
    x = boat1[FEATURES].copy()
    x.columns = FEATURES

    meta["actual_boat1_win"] = y.values
    return x, y, encoders, meta


def suggest_classifier_params(trial: optuna.Trial) -> dict:
    return {
        **FIXED_CLASSIFIER_PARAMS,
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def build_classifier(params: dict) -> LGBMClassifier:
    return LGBMClassifier(**params)


def create_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
):
    def objective(trial: optuna.Trial) -> float:
        clf = build_classifier(suggest_classifier_params(trial))
        clf.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )
        y_proba = clf.predict_proba(x_valid)[:, 1]
        return roc_auc_score(y_valid, y_proba)

    return objective


def optimize_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[LGBMClassifier, dict, optuna.Study]:
    print(f"  Optuna 最適化開始（{N_TRIALS} trials, 検証 ROC-AUC）...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        create_objective(x_train, y_train, x_valid, y_valid),
        n_trials=N_TRIALS,
    )
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best valid ROC-AUC: {study.best_value:.6f}")

    best_params = {**FIXED_CLASSIFIER_PARAMS, **study.best_params}
    clf = build_classifier(best_params)
    print("\n  最適パラメータで LightGBM Classifier 学習...")
    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[early_stopping(50), log_evaluation(50)],
    )
    return clf, best_params, study


def evaluate_classifier(
    clf: LGBMClassifier,
    x: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> dict:
    y_pred = clf.predict(x)
    y_proba = clf.predict_proba(x)[:, 1]

    metrics = {
        "label": label,
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)) if y.nunique() > 1 else None,
    }

    print(f"\n=== 1号艇勝利予測 ({label}) ===")
    print(f"  サンプル数: {metrics['n_samples']}")
    print(f"  1着=1号艇率: {metrics['positive_rate']:.2%}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print("  混同行列 [実際\\予測]  0    1")
    print(f"    0 (1号艇非1着): {cm[0, 0]:5d} {cm[0, 1]:5d}")
    print(f"    1 (1号艇1着):   {cm[1, 0]:5d} {cm[1, 1]:5d}")

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def save_confusion_matrix(cm: list[list[int]], output_csv: Path, output_png: Path) -> None:
    cm_df = pd.DataFrame(
        cm,
        index=["実際: 1号艇非1着", "実際: 1号艇1着"],
        columns=["予測: 0", "予測: 1"],
    )
    cm_df.to_csv(output_csv, encoding="UTF-8-sig")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["予測: 0", "予測: 1"])
    ax.set_yticks([0, 1], labels=["実際: 非1着", "実際: 1着"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    ax.set_title("混同行列（2024年検証）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  混同行列CSV: {output_csv}")
    print(f"  混同行列PNG: {output_png}")


def save_predictions(
    clf: LGBMClassifier,
    x: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    output_path: Path,
) -> None:
    proba = clf.predict_proba(x)[:, 1]
    pred = clf.predict(x)
    out = meta.copy()
    out["P(1号艇1着)"] = proba
    out["予測"] = pred
    out["actual_boat1_win"] = y.values
    out["的中"] = pred == y.values
    out.to_csv(output_path, index=False, encoding="UTF-8-sig")
    print(f"  予測結果CSV: {output_path}")


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    df = load_data()
    train_df, valid_df = split_by_year(df)
    print(
        f"  時系列分割: 学習 {TRAIN_YEAR_END}年まで "
        f"/ 検証 {VALID_YEAR}年（ランダム分割なし）"
    )
    print(f"  学習レース: {train_df.drop_duplicates(RACE_KEY).shape[0]}")
    print(f"  検証レース: {valid_df.drop_duplicates(RACE_KEY).shape[0]}")

    x_train, y_train, encoders, _ = prepare_race_samples(train_df, fit_encoders=True)
    x_valid, y_valid, encoders, valid_meta = prepare_race_samples(
        valid_df, encoders=encoders, fit_encoders=False,
    )

    clf, best_params, study = optimize_classifier(x_train, y_train, x_valid, y_valid)

    train_metrics = evaluate_classifier(clf, x_train, y_train, f"学習 {TRAIN_YEAR_END}年まで")
    valid_metrics = evaluate_classifier(clf, x_valid, y_valid, f"検証 {VALID_YEAR}年")

    save_confusion_matrix(
        valid_metrics["confusion_matrix"],
        CONFUSION_MATRIX_CSV_PATH,
        CONFUSION_MATRIX_PNG_PATH,
    )
    save_predictions(clf, x_valid, y_valid, valid_meta, PREDICTIONS_CSV_PATH)

    summary = {
        "model": "1号艇勝利予測",
        "algorithm": "LightGBM Classifier",
        "target": "P(1号艇が1着)",
        "train_years": f"2014-{TRAIN_YEAR_END}",
        "valid_year": VALID_YEAR,
        "features": FEATURES,
        "best_value": study.best_value,
        "best_metric": "valid ROC-AUC",
        "best_params": study.best_params,
        "classifier_params": best_params,
        "n_trials": N_TRIALS,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(OPTUNA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_metric": "valid ROC-AUC",
                "best_params": study.best_params,
                "n_trials": N_TRIALS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    clf.booster_.save_model(str(MODEL_PATH))
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    print("\nSHAP分析")
    x_sample = x_train.sample(min(SHAP_SAMPLE_SIZE, len(x_train)), random_state=42)
    run_shap_analysis(
        clf.booster_,
        x_sample,
        SHAP_BEESWARM_PNG_PATH,
        SHAP_WATERFALL_PNG_PATH,
        title="1号艇勝利予測モデル",
    )

    print(f"\n保存完了:")
    print(f"  {MODEL_PATH}")
    print(f"  {CLASSIFIER_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"  {METRICS_PATH}")
    print(f"  {OPTUNA_SUMMARY_PATH}")
    print(f"  {CONFUSION_MATRIX_CSV_PATH}")
    print(f"  {CONFUSION_MATRIX_PNG_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {SHAP_BEESWARM_PNG_PATH}")
    print(f"  {SHAP_WATERFALL_PNG_PATH}")


if __name__ == "__main__":
    main()
