"""
3連単予想用オッズ予測モデル（組み合わせ別 LightGBM 回帰）

- データ: レース/選手/気象 CSV + オッズデータ/丸亀_*_3連単オッズ.csv
- 各3連単組み合わせ（120通り）ごとに独立した回帰モデル
- 教師データ: 各レースの的中3連単とその最終オッズ（既存CSV）
- 特徴量: 締切前に取得可能な情報のみ（着順・オッズは入力に含めない）
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from train_race_ranker import RACE_KEY, TRAIN_YEARS, load_race_data
from trifecta_training_utils import UNKNOWN_CATEGORY_TOKEN, setup_japanese_font

# ---------------------------------------------------------------------------
# パス・定数
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ODDS_DATA_DIR = BASE_DIR / "オッズデータ"
TRAIN_ODDS_PATH = ODDS_DATA_DIR / "丸亀学習用_3連単オッズ.csv"
TEST_ODDS_PATH = ODDS_DATA_DIR / "丸亀テスト用_3連単オッズ.csv"
TEST_YEARS = range(2025, 2026)

MODEL_DIR = BASE_DIR / "models" / "3連単オッズ予想"
COMBO_MODEL_DIR = MODEL_DIR / "combinations"
OUTPUT_DIR = MODEL_DIR / "学習"
TEST_OUTPUT_DIR = MODEL_DIR / "テスト"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
COMBO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_PATH = MODEL_DIR / "encoders.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"
EVALUATION_CSV_PATH = OUTPUT_DIR / "評価結果.csv"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "予測結果.csv"

ALL_COMBINATIONS: list[str] = [
    f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)
]

REFERENCE_DATE = pd.Timestamp("2014-01-01")
VALID_YEAR = 2024
MIN_TRAIN_SAMPLES = 20
NUM_BOOST_ROUND = 200
EARLY_STOPPING_ROUNDS = 50
RANDOM_SEED = 42

RACE_LEVEL_COLS = ["開催経過日", "風速", "波高"]
# 特徴量に含めない列
EXCLUDE_FROM_FEATURES = {"選手名", "レース", "開催日", "日目", "算出期間自", "算出期間至", "着", "3連単", "3連単オッズ"}
BOAT_ID_COLS = ["登番", "モーター", "ボート", "級"]
CATEGORICAL_COLS = (
    ["天気", "風向"]
    + [f"{prefix}_{col}" for prefix in ("pos1", "pos2", "pos3") for col in BOAT_ID_COLS]
)
BOAT_STAT_COLS = [
    "展示", "勝率", "2連率", "3連率", "当地勝率", "当地2連率", "当地3連率",
    "モーター勝率", "モーター2連率", "モーター3連率",
    "ボート勝率", "ボート2連率", "ボート3連率", "平均スタートタイミング",
    "1コース複勝率", "2コース複勝率", "3コース複勝率",
    "4コース複勝率", "5コース複勝率", "6コース複勝率", "体重",
]
RELATIVE_BASE_COLS = [
    "展示", "勝率", "2連率", "3連率", "当地勝率",
    "モーター勝率", "ボート勝率", "平均スタートタイミング",
]

BASE_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": RANDOM_SEED,
    "feature_pre_filter": False,
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_data_in_leaf": 15,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
}


def build_feature_names() -> list[str]:
    names = list(RACE_LEVEL_COLS) + ["天気", "風向"]
    for prefix in ("pos1", "pos2", "pos3"):
        names.append(f"{prefix}_艇")
        names.extend(f"{prefix}_{col}" for col in BOAT_ID_COLS)
        for col in BOAT_STAT_COLS:
            names.append(f"{prefix}_{col}")
        for col in RELATIVE_BASE_COLS:
            names.extend([f"{prefix}_{col}_レース内順位", f"{prefix}_{col}_レース内差"])
    return names


FEATURES = build_feature_names()


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------
def load_odds_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"オッズCSVが見つかりません: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = [*RACE_KEY, "3連単", "3連単オッズ"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"オッズCSVに不足列があります: {sorted(missing)}")
    df = df[required].copy()
    df["3連単オッズ"] = pd.to_numeric(df["3連単オッズ"], errors="coerce")
    df = df.dropna(subset=["3連単オッズ"])
    df = df.loc[df["3連単オッズ"] > 0]
    df["日目"] = df["日目"].astype(int)
    df["レース"] = df["レース"].astype(int)
    return df


def load_race_frame(years: range | list[int]) -> pd.DataFrame:
    df = load_race_data(years)
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    df = df.loc[sizes == 6].copy()
    df["__year"] = pd.to_datetime(df["開催日"]).dt.year
    return df


def subset_years(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    return df.loc[df["__year"].isin(years)].copy()


# ---------------------------------------------------------------------------
# 特徴量
# ---------------------------------------------------------------------------
def parse_combination(combo: str) -> tuple[int, int, int]:
    parts = [int(x) for x in combo.split("-")]
    if len(parts) != 3 or len(set(parts)) != 3 or not all(1 <= p <= 6 for p in parts):
        raise ValueError(f"不正な3連単: {combo}")
    return parts[0], parts[1], parts[2]


def _prepare_race(race_df: pd.DataFrame) -> pd.DataFrame:
    work = race_df.sort_values("艇").copy()
    work["開催経過日"] = (pd.to_datetime(work["開催日"]) - REFERENCE_DATE).dt.days
    for col in RELATIVE_BASE_COLS:
        vals = pd.to_numeric(work.get(col, 0), errors="coerce").fillna(0.0)
        work[f"{col}_レース内順位"] = vals.rank(method="min")
        work[f"{col}_レース内差"] = vals - vals.mean()
    return work


def _boat_features(boat_row: pd.Series, prefix: str) -> dict:
    row = {f"{prefix}_艇": int(boat_row["艇"])}
    for col in BOAT_ID_COLS:
        row[f"{prefix}_{col}"] = str(boat_row.get(col, ""))
    for col in BOAT_STAT_COLS:
        row[f"{prefix}_{col}"] = float(pd.to_numeric(boat_row.get(col, 0), errors="coerce") or 0.0)
    for col in RELATIVE_BASE_COLS:
        row[f"{prefix}_{col}_レース内順位"] = float(boat_row.get(f"{col}_レース内順位", 0.0))
        row[f"{prefix}_{col}_レース内差"] = float(boat_row.get(f"{col}_レース内差", 0.0))
    return row


def build_combo_row(race_df: pd.DataFrame, combo: str) -> dict | None:
    if len(race_df) != 6:
        return None
    b1, b2, b3 = parse_combination(combo)
    work = _prepare_race(race_df)
    boats = {int(r["艇"]): r for _, r in work.iterrows()}
    if not all(b in boats for b in (b1, b2, b3)):
        return None

    meta = work.iloc[0]
    row = {
        "開催日": meta["開催日"],
        "日目": int(meta["日目"]),
        "レース": int(meta["レース"]),
        "3連単": combo,
        "天気": meta.get("天気", ""),
        "風向": meta.get("風向", ""),
        "風速": float(pd.to_numeric(meta.get("風速", 0), errors="coerce") or 0.0),
        "波高": float(pd.to_numeric(meta.get("波高", 0), errors="coerce") or 0.0),
        "開催経過日": int(meta["開催経過日"]),
    }
    row.update(_boat_features(boats[b1], "pos1"))
    row.update(_boat_features(boats[b2], "pos2"))
    row.update(_boat_features(boats[b3], "pos3"))
    return row


def encode_categoricals(
    frame: pd.DataFrame,
    encoders: dict,
    *,
    fit: bool,
    categorical_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()
    for col in categorical_cols or CATEGORICAL_COLS:
        if col not in out.columns:
            continue
        encoders.setdefault(col, {})
        values = out[col].astype(str)
        if fit:
            next_idx = max(encoders[col].values(), default=-1) + 1
            for value in values.unique():
                if value not in encoders[col]:
                    encoders[col][value] = next_idx
                    next_idx += 1
        mapped = values.map(encoders[col])
        if mapped.isna().any():
            token = encoders[col].setdefault(
                UNKNOWN_CATEGORY_TOKEN,
                max(encoders[col].values(), default=-1) + 1,
            )
            mapped = mapped.fillna(token)
        out[col] = mapped.astype(int)
    return out, encoders


def build_labeled_frame(
    race_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    encoders: dict | None = None,
    *,
    fit_encoders: bool = True,
) -> tuple[pd.DataFrame, dict]:
    encoders = encoders or {}
    race_map = {
        tuple(k): g for k, g in race_df.groupby(RACE_KEY, sort=False)
    }
    rows: list[dict] = []
    for _, odds_row in odds_df.iterrows():
        key = (odds_row["開催日"], int(odds_row["日目"]), int(odds_row["レース"]))
        race = race_map.get(key)
        if race is None:
            continue
        combo = str(odds_row["3連単"])
        feat = build_combo_row(race, combo)
        if feat is None:
            continue
        feat["3連単オッズ"] = float(odds_row["3連単オッズ"])
        rows.append(feat)

    if not rows:
        raise ValueError("学習用のラベル付きデータを作成できませんでした")

    frame = pd.DataFrame(rows)
    for col in FEATURES:
        if col not in frame.columns:
            frame[col] = "" if col in CATEGORICAL_COLS else 0.0
    for col in FEATURES:
        if col not in CATEGORICAL_COLS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame, encoders = encode_categoricals(frame, encoders, fit=fit_encoders)
    return frame, encoders


def categorical_cols_for(feature_names: list[str], encoders: dict) -> list[str]:
    return [col for col in feature_names if col in encoders]


def to_feature_matrix(
    frame: pd.DataFrame,
    feature_names: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    feature_names = feature_names or FEATURES
    if categorical_cols is None:
        categorical_cols = [c for c in CATEGORICAL_COLS if c in feature_names]
    x = frame.reindex(columns=feature_names, fill_value=0)
    for col in feature_names:
        if col not in categorical_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    return x


# ---------------------------------------------------------------------------
# 学習・推論
# ---------------------------------------------------------------------------
def train_combo_models(
    labeled: pd.DataFrame,
    *,
    valid_frame: pd.DataFrame | None = None,
) -> dict[str, lgb.Booster]:
    models: dict[str, lgb.Booster] = {}
    for combo in ALL_COMBINATIONS:
        train_part = labeled.loc[labeled["3連単"] == combo]
        if len(train_part) < MIN_TRAIN_SAMPLES:
            continue
        x_train = to_feature_matrix(train_part)
        y_train = train_part["3連単オッズ"].to_numpy(dtype=float)
        train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)

        valid_set = None
        callbacks = [lgb.log_evaluation(0)]
        if valid_frame is not None:
            valid_part = valid_frame.loc[valid_frame["3連単"] == combo]
            if len(valid_part) >= 5:
                x_valid = to_feature_matrix(valid_part)
                y_valid = valid_part["3連単オッズ"].to_numpy(dtype=float)
                valid_set = lgb.Dataset(x_valid, label=y_valid, free_raw_data=False)
                callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))

        models[combo] = lgb.train(
            BASE_LGBM_PARAMS,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_set] if valid_set is not None else None,
            valid_names=["valid"] if valid_set is not None else None,
            callbacks=callbacks,
        )
    if not models:
        raise RuntimeError("学習できた組み合わせモデルがありません")
    return models


def predict_combo(
    model: lgb.Booster,
    row: pd.DataFrame,
    feature_names: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> float:
    x = to_feature_matrix(row, feature_names, categorical_cols)
    return max(float(model.predict(x)[0]), 1.0)


def predict_race_odds(
    models: dict[str, lgb.Booster],
    encoders: dict,
    race_df: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    feature_names = feature_names or FEATURES
    categorical_cols = categorical_cols_for(feature_names, encoders)
    rows: list[dict] = []
    for combo in ALL_COMBINATIONS:
        if combo not in models:
            continue
        feat = build_combo_row(race_df, combo)
        if feat is None:
            continue
        frame = pd.DataFrame([feat])
        for col in feature_names:
            if col not in frame.columns:
                frame[col] = "" if col in categorical_cols else 0.0
            elif col not in categorical_cols:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        frame, _ = encode_categoricals(frame, encoders, fit=False, categorical_cols=categorical_cols)
        rows.append({
            **{k: feat[k] for k in RACE_KEY},
            "3連単": combo,
            "予測オッズ": predict_combo(models[combo], frame, feature_names, categorical_cols),
        })
    return pd.DataFrame(rows)


def predict_odds_bulk(
    race_df: pd.DataFrame,
    models: dict[str, lgb.Booster],
    encoders: dict,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """全レース×全組み合わせの予測オッズを一括算出"""
    feature_names = feature_names or FEATURES
    categorical_cols = categorical_cols_for(feature_names, encoders)
    rows: list[dict] = []
    for _, race in race_df.groupby(RACE_KEY, sort=False):
        if len(race) != 6:
            continue
        for combo in ALL_COMBINATIONS:
            if combo not in models:
                continue
            feat = build_combo_row(race, combo)
            if feat is not None:
                rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=[*RACE_KEY, "3連単", "予測オッズ"])

    frame = pd.DataFrame(rows)
    for col in feature_names:
        if col not in frame.columns:
            frame[col] = "" if col in categorical_cols else 0.0
        elif col not in categorical_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame, _ = encode_categoricals(frame, encoders, fit=False, categorical_cols=categorical_cols)
    x = to_feature_matrix(frame, feature_names, categorical_cols)

    preds = np.zeros(len(frame), dtype=float)
    for combo, model in models.items():
        mask = frame["3連単"].values == combo
        if not mask.any():
            continue
        preds[mask] = np.maximum(model.predict(x.loc[mask]), 1.0)
    frame["予測オッズ"] = preds
    return frame[[*RACE_KEY, "3連単", "予測オッズ"]]


def predict_labeled_races(
    models: dict[str, lgb.Booster],
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    x = to_feature_matrix(labeled)
    preds: list[float] = []
    for i, (_, row) in enumerate(labeled.iterrows()):
        model = models.get(row["3連単"])
        if model is None:
            preds.append(np.nan)
        else:
            preds.append(max(float(model.predict(x.iloc[[i]])[0]), 1.0))
    out = labeled[[*RACE_KEY, "3連単", "3連単オッズ"]].copy()
    out["予測オッズ"] = preds
    return out.dropna(subset=["予測オッズ"])


# ---------------------------------------------------------------------------
# 評価・保存
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "n_samples": len(y_true)}


def print_metrics(metrics: dict, label: str) -> None:
    print(f"\n=== 3連単オッズ評価 ({label}) ===")
    print(f"  サンプル数: {metrics['n_samples']}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2%}")
    print(f"  R²:   {metrics['r2']:.4f}")


def evaluate_and_save(
    pred_df: pd.DataFrame,
    label: str,
    metrics_path: Path,
    predictions_path: Path | None = None,
) -> dict:
    metrics = compute_metrics(
        pred_df["3連単オッズ"].to_numpy(),
        pred_df["予測オッズ"].to_numpy(),
    )
    print_metrics(metrics, label)
    row = {"label": label, **metrics}
    pd.DataFrame([row]).to_csv(metrics_path, index=False, encoding="UTF-8-sig")
    if predictions_path is not None:
        pred_df.to_csv(predictions_path, index=False, encoding="UTF-8-sig")
    return row


def save_models(models: dict[str, lgb.Booster], encoders: dict, config: dict) -> None:
    COMBO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for combo, model in models.items():
        model.save_model(str(COMBO_MODEL_DIR / f"{combo}.txt"))
    joblib.dump(
        {"encoders": encoders, "features": FEATURES, "categorical_cols": CATEGORICAL_COLS},
        ENCODER_PATH,
    )
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_models() -> tuple[dict[str, lgb.Booster], dict, list[str]]:
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {ENCODER_PATH}")
    payload = joblib.load(ENCODER_PATH)
    encoders = payload["encoders"]
    features = payload.get("features") or FEATURES
    if not features:
        raise ValueError(f"特徴量リストが空です: {ENCODER_PATH}")
    models: dict[str, lgb.Booster] = {}
    for combo in ALL_COMBINATIONS:
        path = COMBO_MODEL_DIR / f"{combo}.txt"
        if path.exists():
            models[combo] = lgb.Booster(model_file=str(path))
    if not models:
        raise FileNotFoundError(f"組み合わせモデルが見つかりません: {COMBO_MODEL_DIR}")
    return models, encoders, features


def train_model() -> dict:
    print("データ読み込み: 2014〜2024年（レース/選手/気象）")
    race_df = load_race_frame(TRAIN_YEARS)
    print(f"  行数: {len(race_df)} / レース数: {len(race_df) // 6}")

    print(f"オッズデータ: {TRAIN_ODDS_PATH.name}")
    odds_df = load_odds_csv(TRAIN_ODDS_PATH)
    print(f"  オッズ行数: {len(odds_df)} / レース数: {len(odds_df)}")

    train_races = subset_years(race_df, [y for y in range(2014, 2025) if y != VALID_YEAR])
    valid_races = subset_years(race_df, [VALID_YEAR])
    train_odds = odds_df.merge(train_races[RACE_KEY].drop_duplicates(), on=RACE_KEY, how="inner")
    valid_odds = odds_df.merge(valid_races[RACE_KEY].drop_duplicates(), on=RACE_KEY, how="inner")

    print(f"\n  学習: 〜{VALID_YEAR - 1}年 / 検証: {VALID_YEAR}年")
    print("  特徴量生成...")
    train_frame, encoders = build_labeled_frame(train_races, train_odds, fit_encoders=True)
    valid_frame, _ = build_labeled_frame(valid_races, valid_odds, encoders, fit_encoders=False)

    print(f"  学習サンプル: {len(train_frame)} / 検証サンプル: {len(valid_frame)}")
    print(f"  組み合わせ別モデル学習（最大{len(ALL_COMBINATIONS)}本）...")
    models = train_combo_models(train_frame, valid_frame=valid_frame)
    print(f"  学習完了: {len(models)} 本")

    pred_valid = predict_labeled_races(models, valid_frame)
    evaluate_and_save(pred_valid, f"検証{VALID_YEAR}", EVALUATION_CSV_PATH, PREDICTIONS_CSV_PATH)

    print("\n  全期間で再学習...")
    full_frame, encoders = build_labeled_frame(race_df, odds_df, fit_encoders=True)
    final_models = train_combo_models(full_frame)

    config = {
        "model": "3連単オッズ予想",
        "train_years": list(TRAIN_YEARS),
        "valid_year": VALID_YEAR,
        "n_combinations_trained": len(final_models),
        "features": FEATURES,
        "random_seed": RANDOM_SEED,
        "train_samples": len(full_frame),
    }
    save_models(final_models, encoders, config)
    return config


def main() -> None:
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    config = train_model()
    print("\n保存完了:")
    print(f"  {COMBO_MODEL_DIR} ({config['n_combinations_trained']} 本)")
    print(f"  {ENCODER_PATH}")
    print(f"  {CONFIG_PATH}")
    print(f"  {EVALUATION_CSV_PATH}")
    print(f"  {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
