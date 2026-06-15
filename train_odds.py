"""
丸亀3連単の最終オッズを予測する LightGBM 回帰モデルを学習する。

1着1号艇予想 / 1着1号艇以外予想 と同じ構想:
  - 3連単の1着が1号艇のレース → 1着1号艇用モデル
  - 3連単の1着が1号艇以外のレース → 1着1号艇以外用モデル
  - レースデータ + 選手データの艇別特徴量から、3連単組合せのオッズを回帰

使い方:
    python train_odds.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
ODDS_PATH = BASE_DIR / "オッズデータ" / "丸亀学習用_3連単オッズ.csv"
RACE_PATH = BASE_DIR / "レースデータ" / "丸亀学習用_レースデータ.csv"
PLAYER_PATH = BASE_DIR / "レースデータ" / "丸亀学習用_選手データ.csv"
OUTPUT_ROOT = BASE_DIR / "models" / "3連単オッズ予想"

MODEL_CONFIGS = [
    {
        "name": "1着1号艇",
        "output_dir": OUTPUT_ROOT / "1着1号艇",
        "filter_fn": lambda first: first == 1,
        "boat_categorical": ["2着艇", "3着艇"],
    },
    {
        "name": "1着1号艇以外",
        "output_dir": OUTPUT_ROOT / "1着1号艇以外",
        "filter_fn": lambda first: first != 1,
        "boat_categorical": ["1着艇", "2着艇", "3着艇"],
    },
]

RACE_KEY = ["開催日", "日目", "レース"]
RACE_ROW_KEY = ["開催日", "日目", "レース", "艇"]
PLAYER_META_COLS = ["名前漢字", "算出期間自", "算出期間至"]
RACE_EXCLUDE_COLUMNS = {"レース", "着", "選手名", "日目", "開催日", "登番", "モーター", "ボート", "艇"}
RACE_BASE_FEATURES = ["展示", "展示順位", "展示差", "風速", "波高", "天気", "風向"]
CATEGORICAL = ["天気", "風向", "級"]
N_TRIALS = 50
VALID_RATIO = 0.15

FEATURES: list[str] = []
POSITION_PREFIXES = ["1着", "2着", "3着"]

BASE_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": 42,
    "feature_pre_filter": False,
}


def configure_features(player_df: pd.DataFrame) -> None:
    global FEATURES
    player_features = [
        c
        for c in player_df.columns
        if c not in PLAYER_META_COLS
        and c != "登番"
        and c not in RACE_EXCLUDE_COLUMNS
    ]
    FEATURES = RACE_BASE_FEATURES + player_features


def merge_player_data(race_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    race = race_df.copy()
    player = player_df.copy()
    race["開催日"] = pd.to_datetime(race["開催日"])

    if {"算出期間自", "算出期間至"}.issubset(player.columns):
        player["算出期間自"] = pd.to_datetime(player["算出期間自"])
        player["算出期間至"] = pd.to_datetime(player["算出期間至"])
        player_cols = [c for c in player.columns if c != "登番"]
        merged = race.merge(player, on="登番", how="left")
        period_match = (
            merged["算出期間自"].notna()
            & (merged["開催日"] >= merged["算出期間自"])
            & (merged["開催日"] <= merged["算出期間至"])
        )
        matched = (
            merged.loc[period_match, RACE_ROW_KEY + player_cols]
            .drop_duplicates(RACE_ROW_KEY)
        )
        out = race.merge(matched, on=RACE_ROW_KEY, how="left")
    else:
        player = player.drop_duplicates(subset=["登番"], keep="last")
        player_cols = [c for c in player.columns if c != "登番"]
        out = race.merge(player, on="登番", how="left")

    return out.drop(columns=[c for c in PLAYER_META_COLS if c in out.columns])


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    return df.loc[sizes == 6].copy()


def load_odds() -> pd.DataFrame:
    df = pd.read_csv(ODDS_PATH)
    df["開催日"] = pd.to_datetime(df["開催日"]).dt.strftime("%Y-%m-%d")
    boats = df["3連単"].str.split("-", expand=True)
    df["1着艇"] = boats[0].astype(int)
    df["2着艇"] = boats[1].astype(int)
    df["3着艇"] = boats[2].astype(int)
    return df


def load_race_player_data() -> pd.DataFrame:
    race_df = pd.read_csv(RACE_PATH)
    player_df = pd.read_csv(PLAYER_PATH)
    configure_features(player_df)
    df = merge_player_data(race_df, player_df)
    df = filter_complete_races(df)
    df["開催日"] = pd.to_datetime(df["開催日"]).dt.strftime("%Y-%m-%d")
    return df


def prepare_race_features(race_df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = race_df.copy()
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .map(encoders.get(col, {}))
                .fillna(-1)
                .astype(int)
            )
    return df


def fit_encoders(race_df: pd.DataFrame) -> dict:
    encoders: dict[str, dict] = {}
    for col in CATEGORICAL:
        if col in race_df.columns:
            encoders[col] = {v: i for i, v in enumerate(race_df[col].astype(str).unique())}
    return encoders


def fit_boat_encoders(odds_df: pd.DataFrame, boat_cols: list[str]) -> dict[str, dict]:
    encoders: dict[str, dict] = {}
    for col in boat_cols:
        encoders[col] = {v: i for i, v in enumerate(odds_df[col].astype(str).unique())}
    return encoders


def build_model_features(boat_categorical: list[str]) -> list[str]:
    combo_features = [f"{prefix}_{feat}" for prefix in POSITION_PREFIXES for feat in FEATURES]
    return ["レース"] + combo_features + boat_categorical


def combo_to_feature_row(
    race_df: pd.DataFrame,
    combo: tuple[int, int, int],
    boat_categorical: list[str],
    boat_encoders: dict[str, dict],
) -> dict:
    a, b, c = combo
    boats = {1: a, 2: b, 3: c}
    row: dict = {"レース": int(race_df["レース"].iloc[0])}

    for pos, prefix in enumerate(POSITION_PREFIXES, start=1):
        boat = boats[pos]
        boat_row = race_df.loc[race_df["艇"].astype(int) == boat]
        if boat_row.empty:
            raise ValueError(f"艇{boat}が見つかりません")
        boat_row = boat_row.iloc[0]
        for feat in FEATURES:
            row[f"{prefix}_{feat}"] = boat_row[feat]

    boat_values = {"1着艇": a, "2着艇": b, "3着艇": c}
    for col in boat_categorical:
        row[col] = boat_encoders[col].get(str(boat_values[col]), -1)

    return row


def build_odds_dataset(
    odds_df: pd.DataFrame,
    race_df: pd.DataFrame,
    feature_encoders: dict,
    boat_encoders: dict,
    boat_categorical: list[str],
    filter_fn: Callable[[int], bool],
) -> pd.DataFrame:
    odds_df = odds_df[odds_df["1着艇"].map(filter_fn)].copy()
    race_df = prepare_race_features(race_df, feature_encoders)
    race_groups = {
        key: grp
        for key, grp in race_df.groupby(RACE_KEY, sort=False)
    }

    rows: list[dict] = []
    for _, odds_row in odds_df.iterrows():
        key = (odds_row["開催日"], odds_row["日目"], odds_row["レース"])
        if key not in race_groups:
            continue

        combo = (
            int(odds_row["1着艇"]),
            int(odds_row["2着艇"]),
            int(odds_row["3着艇"]),
        )
        feat_row = combo_to_feature_row(
            race_groups[key],
            combo,
            boat_categorical,
            boat_encoders,
        )
        feat_row.update({
            "開催日": odds_row["開催日"],
            "日目": odds_row["日目"],
            "レース": odds_row["レース"],
            "3連単": odds_row["3連単"],
            "3連単オッズ": float(odds_row["3連単オッズ"]),
            "target_log_odds": np.log1p(float(odds_row["3連単オッズ"])),
        })
        rows.append(feat_row)

    return pd.DataFrame(rows)


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.sort(df["開催日"].unique())
    split_idx = int(len(dates) * (1 - VALID_RATIO))
    train_dates = set(dates[:split_idx])
    val_dates = set(dates[split_idx:])
    return df[df["開催日"].isin(train_dates)], df[df["開催日"].isin(val_dates)]


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


def evaluate_predictions(y_true, y_pred, label: str) -> dict:
    true_odds = np.expm1(y_true)
    pred_odds = np.expm1(y_pred)
    metrics = {
        "label": label,
        "n_samples": len(y_true),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
        "r2_log": float(r2_score(y_true, y_pred)),
        "mae_odds": float(mean_absolute_error(true_odds, pred_odds)),
        "mape_odds_pct": float(np.mean(np.abs((true_odds - pred_odds) / true_odds)) * 100),
    }
    print(f"\n=== 評価 ({label}) ===")
    print(f"  件数: {metrics['n_samples']}")
    print(f"  RMSE(log): {metrics['rmse_log']:.4f}")
    print(f"  MAE(オッズ倍率): {metrics['mae_odds']:.2f}")
    print(f"  MAPE: {metrics['mape_odds_pct']:.2f}%")
    print(f"  R2(log): {metrics['r2_log']:.4f}")
    return metrics


def train_model(
    config: dict,
    odds_df: pd.DataFrame,
    race_df: pd.DataFrame,
    feature_encoders: dict,
) -> dict:
    name = config["name"]
    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "lgbm_odds_model.txt"
    encoder_path = output_dir / "odds_encoders.pkl"
    optuna_path = output_dir / "最適化結果.json"
    summary_path = output_dir / "学習_評価.json"
    predictions_path = output_dir / "学習_予測結果.csv"

    boat_categorical = config["boat_categorical"]
    model_features = build_model_features(boat_categorical)

    subset_odds = odds_df[odds_df["1着艇"].map(config["filter_fn"])].copy()
    train_odds, val_odds = split_by_date(subset_odds)
    boat_encoders = fit_boat_encoders(train_odds, boat_categorical)

    train_df = build_odds_dataset(
        train_odds, race_df, feature_encoders, boat_encoders, boat_categorical, config["filter_fn"]
    )
    val_df = build_odds_dataset(
        val_odds, race_df, feature_encoders, boat_encoders, boat_categorical, config["filter_fn"]
    )

    print(f"\n=== {name} ===")
    print(f"  学習: {len(train_df)} 件 / 検証: {len(val_df)} 件")
    if len(train_df) < 100:
        raise ValueError(f"{name}: 学習件数が少なすぎます ({len(train_df)})")

    x_train = train_df[model_features]
    y_train = train_df["target_log_odds"]
    x_valid = val_df[model_features]
    y_valid = val_df["target_log_odds"]

    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = suggest_lgbm_params(trial)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        return model.best_score["valid"]["rmse"]

    print(f"  Optuna 最適化開始（{N_TRIALS} trials）...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best valid RMSE(log): {study.best_value:.6f}")

    best_params = study.best_params
    params = {**BASE_LGBM_PARAMS}
    for key in (
        "num_leaves", "max_depth", "learning_rate", "min_data_in_leaf",
        "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2",
    ):
        params[key] = best_params[key]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=best_params["num_boost_round"],
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(50)],
    )

    train_pred = model.predict(x_train)
    valid_pred = model.predict(x_valid)
    train_metrics = evaluate_predictions(y_train, train_pred, f"{name} / 訓練")
    valid_metrics = evaluate_predictions(y_valid, valid_pred, f"{name} / 検証")

    model.save_model(str(model_path))
    joblib.dump(
        {
            "feature_encoders": feature_encoders,
            "boat_encoders": boat_encoders,
            "model_features": model_features,
            "boat_categorical": boat_categorical,
            "features": FEATURES,
        },
        encoder_path,
    )

    pred_df = val_df[RACE_KEY + ["3連単", "3連単オッズ"]].copy()
    pred_df["予測オッズ"] = np.expm1(valid_pred)
    pred_df["予測誤差"] = pred_df["予測オッズ"] - pred_df["3連単オッズ"]
    pred_df.to_csv(predictions_path, index=False, encoding="UTF-8-sig")

    optuna_summary = {
        "model": name,
        "n_train": len(train_df),
        "n_valid": len(val_df),
        "best_value": study.best_value,
        "best_metric": "valid rmse(log)",
        "best_params": study.best_params,
        "n_trials": N_TRIALS,
        "model_features": model_features,
    }
    summary = {
        "model": name,
        "features": model_features,
        "train": train_metrics,
        "valid": valid_metrics,
    }

    with open(optuna_path, "w", encoding="utf-8") as f:
        json.dump(optuna_summary, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  保存: {model_path}")
    print(f"  保存: {encoder_path}")
    return summary


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("データ読み込み...")
    odds_df = load_odds()
    race_df = load_race_player_data()
    print(f"  オッズ: {len(odds_df)} 件")
    print(f"  レースデータ: {len(race_df)} 行 / {len(race_df) // 6} レース")
    print(f"  特徴量数(艇別): {len(FEATURES)}")

    train_odds, _ = split_by_date(odds_df)
    train_race_keys = set(map(tuple, train_odds[RACE_KEY].values))
    train_race_df = race_df[
        race_df[RACE_KEY].apply(tuple, axis=1).isin(train_race_keys)
    ].copy()
    feature_encoders = fit_encoders(train_race_df)

    summaries = []
    for config in MODEL_CONFIGS:
        summaries.append(train_model(config, odds_df, race_df, feature_encoders))

    combined_path = OUTPUT_ROOT / "学習_評価.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"\n保存完了: {combined_path}")


if __name__ == "__main__":
    main()
