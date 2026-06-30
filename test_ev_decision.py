"""
3連単期待値判定 — 2025年バックテスト

保存済み3連単予想モデル + 市場オッズ + ルールベース実行戦略でバックテストする。
- 3連単予想: models/3連単予想/lgbm_race_ranker.txt
- 市場オッズ: オッズデータ/丸亀オッズデータby保管庫_2019_2025.csv
  （現状5分前オッズ未整備のため、欠損時は締切時オッズを代用）
- 払戻オッズ: オッズデータ/丸亀テスト用_3連単オッズ.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from execution_strategy import (
    ExecutionConfig,
    MIN_MARKET_COMBOS,
    RACE_KEY,
    build_payout_map,
    combo_str,
    compute_expected_values,
    load_market_odds,
    load_payout_odds,
    should_bet,
)
from train_race_ranker import (
    CONFIG_PATH as RANKER_CONFIG_PATH,
    ENCODER_PATH as RANKER_ENCODER_PATH,
    FEATURES as RANKER_FEATURES,
    MODEL_PATH as RANKER_MODEL_PATH,
    load_race_data,
    predict_boat_scores,
    predict_race_trifecta,
)
from trifecta_training_utils import get_actual_trifecta, setup_japanese_font

BASE_DIR = Path(__file__).resolve().parent
TEST_YEARS = range(2025, 2026)
MARKET_ODDS_PATH = BASE_DIR / "オッズデータ" / "丸亀オッズデータby保管庫_2019_2025.csv"
PAYOUT_ODDS_PATH = BASE_DIR / "オッズデータ" / "丸亀テスト用_3連単オッズ.csv"

OUTPUT_DIR = BASE_DIR / "models" / "期待値判定" / "テスト"
RACE_RESULTS_CSV = OUTPUT_DIR / "レース判定結果.csv"
SUMMARY_CSV = OUTPUT_DIR / "バックテスト結果.csv"

EXECUTION_CONFIG = ExecutionConfig(top_n=15, bet_yen=100, roi_threshold=1.0)


def load_saved_ranker_model() -> tuple[lgb.Booster, dict, list[str]]:
    """保存済み3連単予想モデルを読み込む"""
    for path in (RANKER_MODEL_PATH, RANKER_ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"3連単予想モデルが見つかりません: {path}\n先に train_race_ranker.py を実行してください。"
            )

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


def load_test_race_data() -> pd.DataFrame:
    print("テストデータ読み込み: 2025年（レース/選手/気象）")
    df = load_race_data(TEST_YEARS)
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    df = df.loc[sizes == 6].copy()
    print(f"  行数: {len(df)} / レース数: {len(df) // 6}")
    return df


def build_market_odds_by_race(market_df: pd.DataFrame) -> dict[tuple, dict[str, float]]:
    """レースキー → {3連単: 市場オッズ}"""
    by_race: dict[tuple, dict[str, float]] = {}
    for key, group in market_df.groupby(RACE_KEY, sort=False):
        if len(group) < MIN_MARKET_COMBOS:
            continue
        by_race[key] = dict(zip(group["3連単"], group["市場オッズ"]))
    return by_race


def simulate_race(
    race_key: tuple,
    race_df: pd.DataFrame,
    race_scores: np.ndarray,
    market_odds_map: dict[str, float],
    payout_map: dict[tuple, tuple[str, float]],
    config: ExecutionConfig,
) -> dict | None:
    actual_combo = combo_str(get_actual_trifecta(race_df))
    actual_combo_label, actual_odds = payout_map.get(race_key, (actual_combo, np.nan))

    prob_map, _ = predict_race_trifecta(race_scores)
    candidates = compute_expected_values(prob_map, market_odds_map)
    participate, top, avg_expected_roi = should_bet(candidates, config)

    base = {
        "開催日": race_key[0],
        "日目": race_key[1],
        "レース": race_key[2],
        "平均期待ROI": avg_expected_roi * 100,
        "実際3連単": actual_combo_label,
        "実際オッズ": actual_odds,
    }

    if candidates.empty:
        return {
            **base,
            "参加判定": "見送り",
            "実際ROI": 0.0,
            "購入金額": 0,
            "払戻金": 0.0,
            "購入点数": 0,
            "購入15点": "",
            "的中有無": 0,
        }

    top_labels = ";".join(top["3連単"].tolist())

    if not participate:
        return {
            **base,
            "参加判定": "見送り",
            "実際ROI": 0.0,
            "購入金額": 0,
            "払戻金": 0.0,
            "購入点数": 0,
            "購入15点": top_labels,
            "的中有無": 0,
        }

    stake = config.bet_yen * len(top)
    bought = set(top["3連単"].tolist())
    hit = actual_combo_label in bought
    payout = config.bet_yen * actual_odds if hit and pd.notna(actual_odds) else 0.0
    actual_roi = (payout / stake * 100) if stake > 0 else 0.0

    return {
        **base,
        "参加判定": "参加",
        "実際ROI": actual_roi,
        "購入金額": stake,
        "払戻金": payout,
        "購入点数": len(top),
        "購入15点": top_labels,
        "的中有無": int(hit),
    }


def run_backtest(
    race_df: pd.DataFrame,
    ranker_model,
    ranker_encoders: dict,
    ranker_features: list[str],
    market_by_race: dict[tuple, dict[str, float]],
    payout_map: dict[tuple, tuple[str, float]],
    config: ExecutionConfig,
) -> pd.DataFrame:
    sorted_df = race_df.sort_values(RACE_KEY + ["艇"])
    scores = predict_boat_scores(
        ranker_model, sorted_df, ranker_encoders, feature_names=ranker_features,
    )

    rows: list[dict] = []
    offset = 0
    for key, race in sorted_df.groupby(RACE_KEY, sort=False):
        if len(race) != 6:
            continue
        if key not in payout_map:
            continue
        market_odds_map = market_by_race.get(key)
        if not market_odds_map:
            continue
        race_scores = scores[offset : offset + 6]
        offset += 6
        result = simulate_race(
            key, race, race_scores, market_odds_map, payout_map, config,
        )
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


def print_summary(result_df: pd.DataFrame) -> dict:
    if result_df.empty:
        summary = {
            "対象レース数": 0,
            "参加レース数": 0,
            "参加率": 0.0,
            "購入総額": 0,
            "払戻総額": 0.0,
            "年間ROI": 0.0,
            "的中レース数": 0,
            "的中率": 0.0,
            "平均購入点数": 0.0,
            "平均期待ROI": 0.0,
        }
        print("\n=== 期待値判定バックテスト（2025年） ===")
        print(f"  対象レース数: {summary['対象レース数']}")
        print(f"  参加レース数: {summary['参加レース数']}")
        print("  （市場オッズ120通りのレースがないため、バックテスト対象なし）")
        return summary

    n_total = len(result_df)
    joined = result_df.loc[result_df["参加判定"] == "参加"]
    n_joined = len(joined)
    total_stake = int(joined["購入金額"].sum())
    total_payout = float(joined["払戻金"].sum())
    n_hit = int(joined["的中有無"].sum())
    annual_roi = (total_payout / total_stake * 100) if total_stake > 0 else 0.0
    avg_expected_roi = float(joined["平均期待ROI"].mean()) if n_joined > 0 else 0.0
    avg_points = float(joined["購入点数"].mean()) if n_joined > 0 else 0.0

    summary = {
        "対象レース数": n_total,
        "参加レース数": n_joined,
        "参加率": n_joined / n_total if n_total else 0.0,
        "購入総額": total_stake,
        "払戻総額": total_payout,
        "年間ROI": annual_roi,
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
    print(f"  年間ROI: {summary['年間ROI']:.2f}%")
    print(f"  的中レース数: {summary['的中レース数']}")
    print(f"  的中率: {summary['的中率']:.2%}")
    print(f"  平均購入点数: {summary['平均購入点数']:.1f} 点")
    print(f"  平均期待ROI: {summary['平均期待ROI']:.2f}%")
    return summary


def main() -> None:
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n保存済みモデル読み込み")
    print("  [3連単予想]")
    ranker_model, ranker_encoders, ranker_features = load_saved_ranker_model()

    race_df = load_test_race_data()

    print("\n市場オッズ読み込み")
    print(f"  パス: {MARKET_ODDS_PATH}")
    print("  オッズ時点: 5分前（欠損時は締切時オッズを代用）")
    market_df = load_market_odds(MARKET_ODDS_PATH, race_schedule_df=race_df)
    market_by_race = build_market_odds_by_race(market_df)
    print(f"  市場オッズ行数: {len(market_df)}")
    print(f"  120通り揃ったレース数: {len(market_by_race)}")

    payout_df = load_payout_odds(PAYOUT_ODDS_PATH)
    payout_map = build_payout_map(payout_df)
    print(f"  払戻オッズ: {len(payout_df)} 行")

    n_race = len(race_df) // 6
    n_payout = sum(
        1 for key, _ in race_df.groupby(RACE_KEY, sort=False) if key in payout_map
    )
    n_market = sum(
        1 for key, _ in race_df.groupby(RACE_KEY, sort=False) if key in market_by_race
    )
    print(f"  払戻データあり: {n_payout} / {n_race} レース")
    print(f"  市場オッズ120通りあり: {n_market} / {n_race} レース")
    if n_market == 0:
        print(
            "  警告: 2025年の市場オッズ（120通り）が未登録です。"
            " ScrapeOrangeBuoyOdds.py で取得後に再実行してください。"
        )

    print("\nバックテスト実行...")
    result_df = run_backtest(
        race_df,
        ranker_model,
        ranker_encoders,
        ranker_features,
        market_by_race,
        payout_map,
        EXECUTION_CONFIG,
    )
    summary = print_summary(result_df)

    result_df.to_csv(RACE_RESULTS_CSV, index=False, encoding="UTF-8-sig")
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False, encoding="UTF-8-sig")

    print("\n保存完了:")
    print(f"  {RACE_RESULTS_CSV}")
    print(f"  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
