"""
3連単期待値判定 — 2025年バックテスト

保存済みモデルを読み込んで推論のみ実行する（学習は行わない）。
- 3連単予想: models/3連単予想/lgbm_race_ranker.txt
- 3連単オッズ予想: models/3連単オッズ予想/combinations/*.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from train_race_ranker import (
    CONFIG_PATH as RANKER_CONFIG_PATH,
    ENCODER_PATH as RANKER_ENCODER_PATH,
    FEATURES as RANKER_FEATURES,
    MODEL_PATH as RANKER_MODEL_PATH,
    RACE_KEY,
    load_race_data,
    predict_boat_scores,
    predict_race_trifecta,
)
from train_trifecta_odds import (
    ALL_COMBINATIONS,
    COMBO_MODEL_DIR as ODDS_COMBO_MODEL_DIR,
    CONFIG_PATH as ODDS_CONFIG_PATH,
    ENCODER_PATH as ODDS_ENCODER_PATH,
    FEATURES as ODDS_FEATURES,
    load_odds_csv,
    predict_odds_bulk,
)
from trifecta_training_utils import get_actual_trifecta, setup_japanese_font

BASE_DIR = Path(__file__).resolve().parent
TEST_YEARS = range(2025, 2026)
TEST_ODDS_PATH = BASE_DIR / "オッズデータ" / "丸亀テスト用_3連単オッズ.csv"

OUTPUT_DIR = BASE_DIR / "models" / "期待値判定" / "テスト"
RACE_RESULTS_CSV = OUTPUT_DIR / "レース判定結果.csv"
SUMMARY_CSV = OUTPUT_DIR / "バックテスト結果.csv"

TOP_N_BETS = 15
BET_YEN = 100
ROI_THRESHOLD = 1.0  # 100%


def load_saved_ranker_model() -> tuple[lgb.Booster, dict, list[str]]:
    """保存済み3連単予想モデル（レース着順ランキング）を読み込む"""
    for path in (RANKER_MODEL_PATH, RANKER_ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"3連単予想モデルが見つかりません: {path}\n先に train_race_ranker.py を実行してください。")

    features = RANKER_FEATURES
    if RANKER_CONFIG_PATH.exists():
        with open(RANKER_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
        features = config.get("features", RANKER_FEATURES)
        print(f"  設定: {RANKER_CONFIG_PATH.name}（学習 {config.get('train_years', '?')}）")

    model = lgb.Booster(model_file=str(RANKER_MODEL_PATH))
    encoders = joblib.load(RANKER_ENCODER_PATH)
    print(f"  モデル: {RANKER_MODEL_PATH}")
    print(f"  エンコーダ: {RANKER_ENCODER_PATH}")
    return model, encoders, features


def load_saved_odds_models() -> tuple[dict[str, lgb.Booster], dict, list[str]]:
    """保存済み3連単オッズ予想モデル（120通り）を読み込む"""
    if not ODDS_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"オッズ予想エンコーダが見つかりません: {ODDS_ENCODER_PATH}\n先に train_trifecta_odds.py を実行してください。"
        )
    if not ODDS_COMBO_MODEL_DIR.exists():
        raise FileNotFoundError(f"オッズ予想モデルディレクトリが見つかりません: {ODDS_COMBO_MODEL_DIR}")

    payload = joblib.load(ODDS_ENCODER_PATH)
    encoders = payload["encoders"]
    features = payload.get("features") or []
    if not features:
        raise ValueError(f"特徴量リストが空です: {ODDS_ENCODER_PATH}")
    if len(features) != len(ODDS_FEATURES):
        print(f"  特徴量: 保存済み {len(features)} 件（コード定義 {len(ODDS_FEATURES)} 件 — 保存済みを使用）")
    else:
        print(f"  特徴量: {len(features)} 件")

    if ODDS_CONFIG_PATH.exists():
        with open(ODDS_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
        print(f"  設定: {ODDS_CONFIG_PATH.name}（{config.get('n_combinations_trained', '?')} 組み合わせ）")

    models: dict[str, lgb.Booster] = {}
    missing: list[str] = []
    for combo in ALL_COMBINATIONS:
        path = ODDS_COMBO_MODEL_DIR / f"{combo}.txt"
        if path.exists():
            models[combo] = lgb.Booster(model_file=str(path))
        else:
            missing.append(combo)

    if not models:
        raise FileNotFoundError(f"組み合わせモデルが1件も見つかりません: {ODDS_COMBO_MODEL_DIR}")
    if missing:
        print(f"  警告: 未保存の組み合わせ {len(missing)} 件（スキップ）")

    print(f"  モデル: {ODDS_COMBO_MODEL_DIR}（{len(models)} 本）")
    print(f"  エンコーダ: {ODDS_ENCODER_PATH}")
    return models, encoders, features


def combo_str(combo: tuple[int, int, int]) -> str:
    return f"{combo[0]}-{combo[1]}-{combo[2]}"


def load_test_race_data() -> pd.DataFrame:
    print("テストデータ読み込み: 2025年（レース/選手/気象）")
    df = load_race_data(TEST_YEARS)
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    df = df.loc[sizes == 6].copy()
    print(f"  行数: {len(df)} / レース数: {len(df) // 6}")
    return df


def build_actual_odds_map(odds_df: pd.DataFrame) -> dict[tuple, tuple[str, float]]:
    mapping: dict[tuple, tuple[str, float]] = {}
    for _, row in odds_df.iterrows():
        key = (row["開催日"], int(row["日目"]), int(row["レース"]))
        mapping[key] = (str(row["3連単"]), float(row["3連単オッズ"]))
    return mapping


def select_top_bets(
    prob_map: dict[tuple[int, int, int], float],
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for combo, prob in prob_map.items():
        label = combo_str(combo)
        match = odds_df.loc[odds_df["3連単"] == label]
        if match.empty:
            continue
        pred_odds = float(match.iloc[0]["予測オッズ"])
        rows.append({
            "3連単": label,
            "予測確率": prob,
            "予測オッズ": pred_odds,
            "期待値": prob * pred_odds,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("期待値", ascending=False)


def simulate_race(
    race_key: tuple,
    race_df: pd.DataFrame,
    race_scores: np.ndarray,
    race_odds_df: pd.DataFrame,
    actual_odds_map: dict[tuple, tuple[str, float]],
) -> dict:
    actual_combo = combo_str(get_actual_trifecta(race_df))
    actual_key = race_key
    actual_combo_label, actual_odds = actual_odds_map.get(actual_key, (actual_combo, np.nan))

    prob_map, _ = predict_race_trifecta(race_scores)
    candidates = select_top_bets(prob_map, race_odds_df)
    if candidates.empty:
        return {
            "開催日": race_key[0],
            "日目": race_key[1],
            "レース": race_key[2],
            "参加判定": "見送り",
            "予測ROI": 0.0,
            "実際ROI": 0.0,
            "購入金額": 0,
            "払戻金": 0.0,
            "購入点数": 0,
            "購入15点": "",
            "的中有無": 0,
            "実際3連単": actual_combo_label,
            "実際オッズ": actual_odds,
        }

    top = candidates.head(TOP_N_BETS)
    expected_roi = float(top["期待値"].sum() / len(top))
    participate = expected_roi >= ROI_THRESHOLD

    if not participate:
        return {
            "開催日": race_key[0],
            "日目": race_key[1],
            "レース": race_key[2],
            "参加判定": "見送り",
            "予測ROI": expected_roi * 100,
            "実際ROI": 0.0,
            "購入金額": 0,
            "払戻金": 0.0,
            "購入点数": 0,
            "購入15点": ";".join(top["3連単"].tolist()),
            "的中有無": 0,
            "実際3連単": actual_combo_label,
            "実際オッズ": actual_odds,
        }

    stake = BET_YEN * len(top)
    bought = set(top["3連単"].tolist())
    hit = actual_combo_label in bought
    payout = BET_YEN * actual_odds if hit and pd.notna(actual_odds) else 0.0
    actual_roi = (payout / stake * 100) if stake > 0 else 0.0

    return {
        "開催日": race_key[0],
        "日目": race_key[1],
        "レース": race_key[2],
        "参加判定": "参加",
        "予測ROI": expected_roi * 100,
        "実際ROI": actual_roi,
        "購入金額": stake,
        "払戻金": payout,
        "購入点数": len(top),
        "購入15点": ";".join(top["3連単"].tolist()),
        "的中有無": int(hit),
        "実際3連単": actual_combo_label,
        "実際オッズ": actual_odds,
    }


def run_backtest(
    race_df: pd.DataFrame,
    ranker_model,
    ranker_encoders: dict,
    ranker_features: list[str],
    bulk_odds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    sorted_df = race_df.sort_values(RACE_KEY + ["艇"])
    scores = predict_boat_scores(
        ranker_model, sorted_df, ranker_encoders, feature_names=ranker_features,
    )
    actual_odds_map = build_actual_odds_map(odds_df)
    odds_by_race = {
        key: group for key, group in bulk_odds_df.groupby(RACE_KEY, sort=False)
    }

    rows: list[dict] = []
    offset = 0
    for key, race in sorted_df.groupby(RACE_KEY, sort=False):
        if len(race) != 6:
            continue
        if key not in actual_odds_map:
            continue
        race_odds = odds_by_race.get(key)
        if race_odds is None or race_odds.empty:
            continue
        race_scores = scores[offset : offset + 6]
        offset += 6
        rows.append(simulate_race(
            key, race, race_scores, race_odds, actual_odds_map,
        ))
    return pd.DataFrame(rows)


def print_summary(result_df: pd.DataFrame) -> dict:
    n_total = len(result_df)
    joined = result_df.loc[result_df["参加判定"] == "参加"]
    n_joined = len(joined)
    total_stake = int(joined["購入金額"].sum())
    total_payout = float(joined["払戻金"].sum())
    n_hit = int(joined["的中有無"].sum())
    annual_roi = (total_payout / total_stake * 100) if total_stake > 0 else 0.0
    avg_expected_roi = float(joined["予測ROI"].mean()) if n_joined > 0 else 0.0
    avg_points = float(joined["購入点数"].mean()) if n_joined > 0 else 0.0

    summary = {
        "対象レース数": n_total,
        "参加レース数": n_joined,
        "参加率": n_joined / n_total if n_total else 0.0,
        "購入総額": total_stake,
        "払戻総額": total_payout,
        "年間回収率": annual_roi,
        "的中レース数": n_hit,
        "的中率": n_hit / n_joined if n_joined else 0.0,
        "平均購入点数": avg_points,
        "平均期待ROI": avg_expected_roi,
    }

    print("\n=== 期待値判定バックテスト（2025年） ===")
    print(f"  対象レース数: {summary['対象レース数']}")
    print(f"  参加レース数: {summary['参加レース数']}")
    print(f"  参加率: {summary['参加率']:.2%}")
    print(f"  購入総額: {summary['購入総額']:,} 円")
    print(f"  払戻総額: {summary['払戻総額']:,.0f} 円")
    print(f"  年間回収率（ROI）: {summary['年間回収率']:.2f}%")
    print(f"  的中レース数: {summary['的中レース数']}")
    print(f"  的中率: {summary['的中率']:.2%}")
    print(f"  平均購入点数: {summary['平均購入点数']:.1f} 点")
    print(f"  1レース当たりの平均期待ROI: {summary['平均期待ROI']:.2f}%")
    return summary


def main() -> None:
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n保存済みモデル読み込み")
    print("  [3連単予想]")
    ranker_model, ranker_encoders, ranker_features = load_saved_ranker_model()
    print("  [3連単オッズ予想]")
    odds_models, odds_encoders, odds_features = load_saved_odds_models()

    race_df = load_test_race_data()
    odds_df = load_odds_csv(TEST_ODDS_PATH)
    print(f"  オッズデータ: {len(odds_df)} 行")

    print("\n予測オッズ一括算出...")
    bulk_odds_df = predict_odds_bulk(race_df, odds_models, odds_encoders, odds_features)
    print(f"  予測行数: {len(bulk_odds_df)}")

    print("\nバックテスト実行...")
    result_df = run_backtest(
        race_df,
        ranker_model,
        ranker_encoders,
        ranker_features,
        bulk_odds_df,
        odds_df,
    )
    summary = print_summary(result_df)

    result_df.to_csv(RACE_RESULTS_CSV, index=False, encoding="UTF-8-sig")
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False, encoding="UTF-8-sig")

    print("\n保存完了:")
    print(f"  {RACE_RESULTS_CSV}")
    print(f"  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
