"""
1号艇勝利予測モデルのテスト（2025年データ）

train_boat1_win.py と同じ前処理・評価ロジック
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
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
    encode_categorical_columns,
    run_shap_analysis,
    setup_japanese_font,
)

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "レースデータ" / "丸亀テスト用_レースデータ.csv"
MODEL_DIR = BASE_DIR / "models" / "1号艇勝利予測"
TEST_OUTPUT_DIR = MODEL_DIR / "テスト"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_PATH = MODEL_DIR / "boat1_win_classifier.pkl"
ENCODER_PATH = MODEL_DIR / "boat1_win_encoders.pkl"
EVALUATION_PATH = TEST_OUTPUT_DIR / "評価結果.json"
CONFUSION_MATRIX_CSV_PATH = TEST_OUTPUT_DIR / "混同行列.csv"
PREDICTIONS_CSV_PATH = TEST_OUTPUT_DIR / "予測結果.csv"
SHAP_BEESWARM_PNG_PATH = TEST_OUTPUT_DIR / "特徴量影響方向.png"
SHAP_WATERFALL_PNG_PATH = TEST_OUTPUT_DIR / "特徴量ウォーターフォール.png"

RACE_KEY = ["開催日", "日目", "レース"]
TARGET_BOAT = 1
SHAP_SAMPLE_SIZE = 2000

FEATURES: list[str] = []


def configure_features(df: pd.DataFrame) -> None:
    global FEATURES
    FEATURES = build_feature_list(df)


def load_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}")
    df = pd.read_csv(RACE_DATA_PATH, low_memory=False)
    configure_features(df)
    df = filter_complete_races(df)
    n_races = df.drop_duplicates(RACE_KEY).shape[0]
    print(f"  対象: {len(df)} 行 / {n_races} レース")
    print(f"  特徴量数: {len(FEATURES)}")
    return df


def load_model_and_encoder() -> tuple[LGBMClassifier, dict]:
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {CLASSIFIER_PATH}\n"
            "先に python train_boat1_win.py を実行してください。"
        )
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {ENCODER_PATH}")
    return joblib.load(CLASSIFIER_PATH), joblib.load(ENCODER_PATH)


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    return df.loc[sizes == 6].copy()


def prepare_race_samples(
    df: pd.DataFrame,
    encoders: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = filter_complete_races(df)
    df = df.sort_values(RACE_KEY).reset_index(drop=True)
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")

    boat1 = df.loc[df["艇"].astype(int) == TARGET_BOAT].copy()
    meta = boat1[RACE_KEY + ["開催日", "着"]].copy()
    boat1 = boat1.drop(columns=["日目", "選手名"], errors="ignore")

    for col in CATEGORICAL_FEATURES:
        if col not in encoders:
            raise KeyError(f"エンコーダに {col} がありません")
    boat1, encoders = encode_categorical_columns(boat1, encoders, update_encoders=False)

    y = (boat1["着"].astype(int) == 1).astype(int)
    x = boat1[FEATURES].copy()
    x.columns = FEATURES
    return x, y, meta


def main():
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    clf, encoders = load_model_and_encoder()
    print(f"  モデル: {CLASSIFIER_PATH.name}")

    df = load_data()
    print(
        f"  期間: {pd.to_datetime(df['開催日']).min().date()} 〜 "
        f"{pd.to_datetime(df['開催日']).max().date()}"
    )

    x, y, meta = prepare_race_samples(df, encoders)
    y_pred = clf.predict(x)
    y_proba = clf.predict_proba(x)[:, 1]

    metrics = {
        "label": "テスト / 1号艇勝利予測",
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)) if y.nunique() > 1 else None,
    }

    print(f"\n=== 1号艇勝利予測 ({metrics['label']}) ===")
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

    pd.DataFrame(
        cm,
        index=["実際: 1号艇非1着", "実際: 1号艇1着"],
        columns=["予測: 0", "予測: 1"],
    ).to_csv(CONFUSION_MATRIX_CSV_PATH, encoding="UTF-8-sig")

    out = meta.copy()
    out["P(1号艇1着)"] = y_proba
    out["予測"] = y_pred
    out["actual_boat1_win"] = y.values
    out["的中"] = y_pred == y.values
    out.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="UTF-8-sig")

    metrics["confusion_matrix"] = cm.tolist()
    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nSHAP分析")
    x_sample = x.sample(min(SHAP_SAMPLE_SIZE, len(x)), random_state=42)
    run_shap_analysis(
        clf.booster_,
        x_sample,
        SHAP_BEESWARM_PNG_PATH,
        SHAP_WATERFALL_PNG_PATH,
        title="1号艇勝利予測モデル（テスト）",
    )

    print(f"\n保存完了:")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {CONFUSION_MATRIX_CSV_PATH}")
    print(f"  {EVALUATION_PATH}")
    print(f"  {SHAP_BEESWARM_PNG_PATH}")
    print(f"  {SHAP_WATERFALL_PNG_PATH}")


if __name__ == "__main__":
    main()
