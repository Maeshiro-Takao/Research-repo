"""
丸亀3連単オッズ予測モデルをテストデータで検証する。

1着1号艇 / 1着1号艇以外 の2モデルに、3連単の1着艇番で振り分けて評価する。

使い方:
    python test_odds.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_odds import (
    MODEL_CONFIGS,
    RACE_KEY,
    build_odds_dataset,
    filter_complete_races,
    fit_encoders,
    merge_player_data,
    configure_features,
)

BASE_DIR = Path(__file__).resolve().parent
ODDS_PATH = BASE_DIR / "オッズデータ" / "丸亀テスト用_3連単オッズ.csv"
RACE_PATH = BASE_DIR / "レースデータ" / "丸亀テスト用_レースデータ.csv"
PLAYER_PATH = BASE_DIR / "レースデータ" / "丸亀テスト用_選手データ.csv"
OUTPUT_ROOT = BASE_DIR / "models" / "3連単オッズ予想"


def load_test_odds() -> pd.DataFrame:
    df = pd.read_csv(ODDS_PATH)
    df["開催日"] = pd.to_datetime(df["開催日"]).dt.strftime("%Y-%m-%d")
    boats = df["3連単"].str.split("-", expand=True)
    df["1着艇"] = boats[0].astype(int)
    df["2着艇"] = boats[1].astype(int)
    df["3着艇"] = boats[2].astype(int)
    return df


def load_test_race_player_data() -> pd.DataFrame:
    race_df = pd.read_csv(RACE_PATH)
    player_df = pd.read_csv(PLAYER_PATH)
    configure_features(player_df)
    df = merge_player_data(race_df, player_df)
    df = filter_complete_races(df)
    df["開催日"] = pd.to_datetime(df["開催日"]).dt.strftime("%Y-%m-%d")
    return df


def load_model_bundle(config: dict) -> tuple[lgb.Booster, dict]:
    model_dir = config["output_dir"]
    model_path = model_dir / "lgbm_odds_model.txt"
    encoder_path = model_dir / "odds_encoders.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {encoder_path}")
    return lgb.Booster(model_file=str(model_path)), joblib.load(encoder_path)


def evaluate_subset(
    name: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    true_odds = np.expm1(y_true)
    pred_odds = np.expm1(y_pred)
    metrics = {
        "label": f"テスト / {name}",
        "n_races": len(df),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
        "r2_log": float(r2_score(y_true, y_pred)),
        "mae_odds": float(mean_absolute_error(true_odds, pred_odds)),
        "mape_odds_pct": float(np.mean(np.abs((true_odds - pred_odds) / true_odds)) * 100),
    }
    print(f"\n=== テスト評価 ({name}) ===")
    print(f"  件数: {metrics['n_races']}")
    print(f"  RMSE(log): {metrics['rmse_log']:.4f}")
    print(f"  MAE(オッズ倍率): {metrics['mae_odds']:.2f}")
    print(f"  MAPE: {metrics['mape_odds_pct']:.2f}%")
    print(f"  R2(log): {metrics['r2_log']:.4f}")
    return metrics


def main():
    odds_df = load_test_odds()
    race_df = load_test_race_player_data()
    print(f"テストオッズ: {len(odds_df)} 件")
    print(f"テストレース: {len(race_df) // 6} レース")

    all_results = []
    all_metrics = []
    y_true_all: list[float] = []
    y_pred_all: list[float] = []

    for config in MODEL_CONFIGS:
        model, bundle = load_model_bundle(config)
        feature_encoders = bundle["feature_encoders"]
        boat_encoders = bundle["boat_encoders"]
        boat_categorical = bundle["boat_categorical"]
        model_features = bundle["model_features"]

        subset = build_odds_dataset(
            odds_df,
            race_df,
            feature_encoders,
            boat_encoders,
            boat_categorical,
            config["filter_fn"],
        )
        if len(subset) == 0:
            print(f"\n警告: {config['name']} のテストデータがありません")
            continue

        x = subset[model_features]
        pred_log = model.predict(x)
        y_true = subset["target_log_odds"].astype(float).values
        y_pred = pred_log.astype(float)

        metrics = evaluate_subset(config["name"], subset, y_true, y_pred)
        all_metrics.append(metrics)
        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())

        result = subset[RACE_KEY + ["3連単", "3連単オッズ"]].copy()
        result["予測オッズ"] = np.expm1(y_pred)
        result["予測誤差"] = result["予測オッズ"] - result["3連単オッズ"]
        result["モデル"] = config["name"]
        all_results.append(result)

        out_dir = config["output_dir"]
        result.to_csv(out_dir / "テスト_予測結果.csv", index=False, encoding="UTF-8-sig")
        with open(out_dir / "テスト_評価.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if y_true_all:
        combined_metrics = evaluate_subset(
            "全体",
            pd.DataFrame({"dummy": range(len(y_true_all))}),
            np.array(y_true_all),
            np.array(y_pred_all),
        )
        all_metrics.append(combined_metrics)

        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_ROOT / "テスト_予測結果.csv", index=False, encoding="UTF-8-sig")
        with open(OUTPUT_ROOT / "テスト_評価.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)

        print(f"\n保存完了:")
        print(f"  {OUTPUT_ROOT / 'テスト_予測結果.csv'}")
        print(f"  {OUTPUT_ROOT / 'テスト_評価.json'}")


if __name__ == "__main__":
    main()
