"""
保存済みモデルで丸亀テスト用_レースデータ.csv / 丸亀テスト用_選手データ.csv を検証する。

使い方:
    python train_trifecta.py   # 先にモデル学習
    python test_trifecta.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

import train_trifecta as tt

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "編集データ" / "丸亀テスト用_レースデータ.csv"
PLAYER_DATA_PATH = BASE_DIR / "編集データ" / "丸亀テスト用_選手データ.csv"
MODEL_PATH = tt.MODEL_DIR / "lgbm_trifecta_model.txt"
ENCODER_PATH = tt.MODEL_DIR / "trifecta_encoders.pkl"
OUTPUT_PATH = tt.MODEL_DIR / "trifecta_test_predictions.csv"


def load_model_and_encoders() -> tuple[lgb.Booster, dict, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {ENCODER_PATH}")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    encoders = joblib.load(ENCODER_PATH)
    features = model.feature_name()
    return model, encoders, features


def load_test_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}, {PLAYER_DATA_PATH.name}")
    race_df = pd.read_csv(RACE_DATA_PATH, low_memory=False)
    player_df = pd.read_csv(PLAYER_DATA_PATH, low_memory=False)
    df = tt.merge_player_data(race_df, player_df)

    matched = df["級"].notna().sum() if "級" in df.columns else 0
    print(f"  全体: {len(df)} 行 / {len(df) // 6} レース")
    print(f"  選手データ結合: {matched} 行 ({matched / len(df):.1%})")
    print(
        f"  期間: {pd.to_datetime(df['開催日']).min().date()} 〜 "
        f"{pd.to_datetime(df['開催日']).max().date()}"
    )
    return df


def prepare_inference(df: pd.DataFrame, encoders: dict, features: list[str]):
    """保存済みエンコーダで推論用特徴量を作成する"""
    df = tt.filter_complete_races(df)
    df = df.sort_values(tt.RACE_KEY).reset_index(drop=True)
    boat_numbers = df["艇"].astype(int).values
    df["展示順位"] = df.groupby(tt.RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(tt.RACE_KEY)["展示"].transform("mean")
    groups = df.groupby(tt.RACE_KEY, sort=False).size().tolist()
    df = df.drop(columns=["日目", "選手名"], errors="ignore")
    for col in tt.CATEGORICAL:
        if col in encoders:
            df[col] = (
                df[col].astype(str).map(encoders[col]).fillna(-1).astype(int)
            )

    for col in features:
        if col not in df.columns:
            df[col] = pd.NA

    x = df[features]
    y = 7 - df["着"].astype(int) if "着" in df.columns else None
    return x, y, groups, encoders, boat_numbers


def patch_train_module(features: list[str]) -> None:
    """train_trifecta の推論処理をテスト用に差し替える"""
    tt.FEATURES = features

    def _prepare(df, encoders=None):
        return prepare_inference(df, encoders or {}, features)

    tt.prepare = _prepare


def main():
    model, encoders, features = load_model_and_encoders()
    patch_train_module(features)
    print(f"  特徴量数: {len(features)}")

    df = load_test_data()
    tt.evaluate_trifecta(model, df, encoders, "テスト")

    print("\n3連単予測を保存...")
    tt.save_trifecta_predictions(model, df, encoders, OUTPUT_PATH)


if __name__ == "__main__":
    main()
