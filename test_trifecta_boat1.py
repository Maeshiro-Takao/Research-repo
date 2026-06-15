from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "レースデータ" / "丸亀テスト用_レースデータ.csv"
PLAYER_DATA_PATH = BASE_DIR / "レースデータ" / "丸亀テスト用_選手データ.csv"
OUTPUT_DIR = BASE_DIR / "models" / "1着1号艇予想"

MODEL_PATH = OUTPUT_DIR / "lgbm_trifecta_boat1_model.txt"
ENCODER_PATH = OUTPUT_DIR / "trifecta_boat1_encoders.pkl"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "テスト_予測結果.csv"
EVALUATION_CSV_PATH = OUTPUT_DIR / "テスト_評価結果.csv"

RACE_KEY = ["開催日", "日目", "レース"]
RACE_ROW_KEY = ["開催日", "日目", "レース", "艇"]
TARGET_BOAT = 1
PLAYER_META_COLS = ["名前漢字", "算出期間自", "算出期間至"]
CATEGORICAL = ["天気", "風向", "級"]
TOP_N_LIST = [1, 3, 5, 10, 30]


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


def load_test_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}, {PLAYER_DATA_PATH.name}")
    race_df = pd.read_csv(RACE_DATA_PATH, low_memory=False)
    player_df = pd.read_csv(PLAYER_DATA_PATH, low_memory=False)
    df = filter_complete_races(merge_player_data(race_df, player_df))
    matched = df["級"].notna().sum() if "級" in df.columns else 0
    print(f"  全体: {len(df)} 行 / {len(df) // 6} レース")
    print(f"  選手データ結合: {matched} 行 ({matched / len(df):.1%})")
    print(
        f"  期間: {pd.to_datetime(df['開催日']).min().date()} 〜 "
        f"{pd.to_datetime(df['開催日']).max().date()}"
    )
    return df


def load_model_and_encoder() -> tuple[lgb.Booster, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"エンコーダが見つかりません: {ENCODER_PATH}")
    return lgb.Booster(model_file=str(MODEL_PATH)), joblib.load(ENCODER_PATH)


def prepare_race(
    race_df: pd.DataFrame,
    features: list[str],
    encoders: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    """2〜6号艇のみを特徴量化する"""
    df = race_df.copy()
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    df = df.loc[df["艇"].astype(int) != TARGET_BOAT].copy()
    boat_numbers = df["艇"].astype(int).values
    for col in CATEGORICAL:
        if col in encoders:
            df[col] = df[col].astype(str).map(encoders[col]).fillna(-1).astype(int)
    for col in features:
        if col not in df.columns:
            df[col] = pd.NA
    return df[features], boat_numbers


def calc_trifecta_probs_boat1_fixed(
    boat_scores: dict[int, float],
) -> dict[tuple[int, int, int], float]:
    boats = sorted(boat_scores)
    max_score = max(boat_scores.values())
    exp_scores = {b: np.exp(boat_scores[b] - max_score) for b in boats}
    total = sum(exp_scores.values())
    results: dict[tuple[int, int, int], float] = {}
    for second in boats:
        p2 = max(exp_scores[second] / total, 1e-12)
        remaining = [b for b in boats if b != second]
        rem_total = sum(exp_scores[b] for b in remaining)
        for third in remaining:
            p3 = max(exp_scores[third] / rem_total, 1e-12)
            results[(TARGET_BOAT, second, third)] = p2 * p3
    return results


def get_actual_trifecta(race_df: pd.DataFrame) -> tuple[int, int, int]:
    top3 = race_df.sort_values("着").head(3)
    return tuple(top3["艇"].astype(int).tolist())  # type: ignore[return-value]


def predict_race_trifecta(
    model: lgb.Booster,
    race_df: pd.DataFrame,
    encoders: dict,
) -> tuple[tuple[int, int, int], list[tuple[tuple[int, int, int], float]]]:
    if len(race_df) != 6:
        raise ValueError("6艇揃ったレースが必要です")

    features = model.feature_name()
    x, boat_numbers = prepare_race(race_df, features, encoders)
    scores = model.predict(x)
    boat_scores = {int(boat): float(score) for boat, score in zip(boat_numbers, scores)}
    ranking = sorted(
        calc_trifecta_probs_boat1_fixed(boat_scores).items(),
        key=lambda x: x[1],
        reverse=True,
    )
    pred_top = ranking[0][0]
    assert pred_top[0] == TARGET_BOAT
    return pred_top, ranking


def main():
    model, encoders = load_model_and_encoder()
    print(f"  特徴量数: {len(model.feature_name())}")

    df = load_test_data()
    hits = {n: 0 for n in TOP_N_LIST}
    first_hits = second_hits = 0
    logloss_sum = 0.0
    n_races = 0
    rows = []

    for key, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        actual = get_actual_trifecta(race_df)
        pred_top, ranking = predict_race_trifecta(model, race_df, encoders)
        prob_map = dict(ranking)
        for n in TOP_N_LIST:
            if actual in [c for c, _ in ranking[:n]]:
                hits[n] += 1
        first_hits += pred_top[0] == actual[0]
        second_hits += pred_top[:2] == actual[:2]
        logloss_sum += -np.log(max(prob_map.get(actual, 1e-12), 1e-12))
        n_races += 1
        rows.append({
            "開催日": key[0],
            "レース": key[2],
            "実際3連単": "-".join(map(str, actual)),
            "予測3連単": "-".join(map(str, pred_top)),
            "予測確率": ranking[0][1],
            "的中": pred_top == actual,
            "TOP10": " / ".join(
                f"{'-'.join(map(str, c))}({p:.4f})" for c, p in ranking[:10]
            ),
        })

    print(f"\n=== 3連単評価 ===")
    print(f"  レース数: {n_races}")
    for n in TOP_N_LIST:
        print(f"  3連単 TOP{n} 的中率: {hits[n] / n_races:.2%}")
    print(f"  1着艇番 的中率: {first_hits / n_races:.2%}")
    print(f"  2連単 的中率:   {second_hits / n_races:.2%}")
    print(f"  3連単 的中率:   {hits[1] / n_races:.2%}")
    print(f"  平均-log確率: {logloss_sum / n_races:.4f}")

    metrics = {
        "label": "テスト / 1着1号艇予想",
        "n_races": n_races,
        "first_hit_rate": first_hits / n_races,
        "exacta_hit_rate": second_hits / n_races,
        "trifecta_top1_rate": hits[1] / n_races,
        "trifecta_avg_neg_log_prob": logloss_sum / n_races,
        **{f"trifecta_top{n}_rate": hits[n] / n_races for n in TOP_N_LIST},
    }
    pd.DataFrame([metrics]).to_csv(EVALUATION_CSV_PATH, index=False, encoding="UTF-8-sig")
    pd.DataFrame(rows).to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="UTF-8-sig")

    print(f"\n保存完了:")
    print(f"  {PREDICTIONS_CSV_PATH}")
    print(f"  {EVALUATION_CSV_PATH}")


if __name__ == "__main__":
    main()
