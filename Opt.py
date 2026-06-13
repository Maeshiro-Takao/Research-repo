import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import lightgbm as lgb
import optuna
import pandas as pd

# ============================================================
# 設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = BASE_DIR / "編集データ" / "丸亀学習用_訓練データ.csv"
VALID_DATA_PATH = BASE_DIR / "編集データ" / "丸亀学習用_検証データ.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RACE_KEY = ["開催日", "日目", "レース"]
EXCLUDE_COLUMNS = {"選手名", "日目", "開催日", "着", "レース"}
RACE_BASE_COLUMNS = {
    "艇", "登番", "モーター", "ボート", "展示", "風速", "波高", "天気", "風向",
}
DERIVED_COLUMNS = ["展示順位", "展示差"]
N_TRIALS = 50

FEATURE_COLUMNS: list[str] = []
CATEGORICAL_FEATURES = ["艇", "登番", "モーター", "ボート", "天気", "風向", "級"]

# ============================================================
# 前処理
# ============================================================

def configure_features(df: pd.DataFrame) -> None:
    """結合済みデータから全特徴量列を設定する"""
    global FEATURE_COLUMNS

    exclude = EXCLUDE_COLUMNS
    player_columns = [
        c for c in df.columns
        if c not in exclude and c not in RACE_BASE_COLUMNS
    ]
    FEATURE_COLUMNS = (
        [c for c in RACE_BASE_COLUMNS if c in df.columns]
        + DERIVED_COLUMNS
        + player_columns
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    return df


def encode_features(df: pd.DataFrame, encoders: Optional[Dict] = None, fit: bool = False):
    encoders = encoders or {}
    encoded = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col not in encoded.columns:
            continue
        if fit:
            encoders[col] = {
                v: i for i, v in enumerate(encoded[col].astype(str).unique())
            }
        encoded[col] = (
            encoded[col].astype(str).map(encoders[col]).fillna(-1).astype(int)
        )

    x = encoded[FEATURE_COLUMNS]
    y = encoded["着"] - 1
    return x, y, encoders


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("選手名", "日目", "開催日", "レース") if c in df.columns]
    return df.drop(columns=cols)


def load_data():
    df_train = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    df_valid = pd.read_csv(VALID_DATA_PATH, low_memory=False)
    configure_features(df_train)

    df_train = drop_unused_columns(add_features(df_train))
    df_valid = drop_unused_columns(add_features(df_valid))

    x_train, y_train, encoders = encode_features(df_train, fit=True)
    x_valid, y_valid, _ = encode_features(df_valid, encoders=encoders, fit=False)

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in FEATURE_COLUMNS],
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in FEATURE_COLUMNS],
        reference=train_set,
        free_raw_data=False,
    )
    return train_set, valid_set, encoders, x_train, y_train, x_valid, y_valid


# ============================================================
# Optuna
# ============================================================

def create_objective(train_set, valid_set):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multiclass",
            "num_class": 6,
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": 42,
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

        num_boost_round = trial.suggest_int("num_boost_round", 100, 500)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
            ],
        )

        return model.best_score["valid"]["multi_logloss"]

    return objective


def train_best_model(train_set, valid_set, best_params: dict) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": 6,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": 42,
        "num_leaves": best_params["num_leaves"],
        "max_depth": best_params["max_depth"],
        "learning_rate": best_params["learning_rate"],
        "min_data_in_leaf": best_params["min_data_in_leaf"],
        "feature_fraction": best_params["feature_fraction"],
        "bagging_fraction": best_params["bagging_fraction"],
        "bagging_freq": best_params["bagging_freq"],
        "lambda_l1": best_params["lambda_l1"],
        "lambda_l2": best_params["lambda_l2"],
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=best_params["num_boost_round"],
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(50),
        ],
    )
    return model


# ============================================================
# メイン
# ============================================================

def main():
    print("データ読み込み...")
    print(f"  {TRAIN_DATA_PATH.name}")
    print(f"  {VALID_DATA_PATH.name}")
    train_set, valid_set, encoders, x_train, y_train, x_valid, y_valid = load_data()
    print(f"  訓練: {len(x_train)} 行")
    print(f"  検証: {len(x_valid)} 行")
    print(f"  特徴量数: {len(FEATURE_COLUMNS)}")

    print(f"\nOptuna 最適化開始（{N_TRIALS} trials）...")
    study = optuna.create_study(direction="minimize")
    study.optimize(create_objective(train_set, valid_set), n_trials=N_TRIALS)

    print("\n=== 最適化結果 ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best valid multi_logloss: {study.best_value:.6f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n最適パラメータで再学習...")
    model = train_best_model(train_set, valid_set, study.best_params)

    model.save_model(str(MODEL_DIR / "lgbm_model.txt"))
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

    summary = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_iteration": model.best_iteration,
        "n_trials": N_TRIALS,
        "features": FEATURE_COLUMNS,
    }
    with open(MODEL_DIR / "optuna_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n保存完了:")
    print(f"  {MODEL_DIR / 'lgbm_model.txt'}")
    print(f"  {MODEL_DIR / 'encoders.pkl'}")
    print(f"  {MODEL_DIR / 'optuna_summary.json'}")


if __name__ == "__main__":
    main()
