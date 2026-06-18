"""
3モデル統合予想

各レースで:
1. 1号艇勝利予測 → P(1号艇1着)
2. 1着1号艇予想 / 1着1号艇以外予想 → それぞれ Plackett-Luce で全3連単確率
3. 確率を合成して統合ランキングを生成

合成式:
  P(1-j-k) = P(1号艇1着) × P_PL_boat1(1-j-k)
  P(i-j-k) = (1-P(1号艇1着)) × P_PL_not(i-j-k | 1着≠1号艇)  （i≠1）

参考としてハードルーティング（0/1振り分け）の的中率も併記する。

使い方:
  python predict_combined.py
  python predict_combined.py --data レースデータ/丸亀学習用_レースデータ.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from trifecta_training_utils import (
    CATEGORICAL_FEATURES,
    build_feature_list,
    encode_categorical_columns,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "レースデータ" / "丸亀テスト用_レースデータ.csv"
OUTPUT_DIR = BASE_DIR / "models" / "統合予想" / "テスト"

WIN_MODEL_DIR = BASE_DIR / "models" / "1号艇勝利予測"
BOAT1_MODEL_DIR = BASE_DIR / "models" / "1着1号艇予想"
NOT_BOAT1_MODEL_DIR = BASE_DIR / "models" / "1着1号艇以外予想"

RACE_KEY = ["開催日", "日目", "レース"]
TARGET_BOAT = 1
TOP_N_LIST = [1, 5, 10, 15, 20]
ROUTE_BOAT1 = "1着1号艇予想"
ROUTE_NOT_BOAT1 = "1着1号艇以外予想"

FEATURES: list[str] = []


def configure_features(df: pd.DataFrame) -> None:
    global FEATURES
    FEATURES = build_feature_list(df)


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    return df.loc[sizes == 6].copy()


def get_actual_trifecta(race_df: pd.DataFrame) -> tuple[int, int, int]:
    top3 = race_df.sort_values("着").head(3)
    return tuple(top3["艇"].astype(int).tolist())  # type: ignore[return-value]


def actual_boat1_win(race_df: pd.DataFrame) -> bool:
    return int(race_df.loc[race_df["着"] == 1, "艇"].iloc[0]) == TARGET_BOAT


def load_models() -> tuple[LGBMClassifier, dict, lgb.Booster, dict, lgb.Booster, dict]:
    win_clf_path = WIN_MODEL_DIR / "boat1_win_classifier.pkl"
    win_enc_path = WIN_MODEL_DIR / "boat1_win_encoders.pkl"
    boat1_model_path = BOAT1_MODEL_DIR / "lgbm_trifecta_boat1_model.txt"
    boat1_enc_path = BOAT1_MODEL_DIR / "trifecta_boat1_encoders.pkl"
    not_boat1_model_path = NOT_BOAT1_MODEL_DIR / "lgbm_trifecta_not_boat1_model.txt"
    not_boat1_enc_path = NOT_BOAT1_MODEL_DIR / "trifecta_not_boat1_encoders.pkl"

    missing = [p for p in (
        win_clf_path, win_enc_path,
        boat1_model_path, boat1_enc_path,
        not_boat1_model_path, not_boat1_enc_path,
    ) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "モデルファイルが見つかりません:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n先に train_boat1_win.py / train_trifecta_boat1.py / "
            "train_trifecta_not_boat1.py を実行してください。"
        )

    return (
        joblib.load(win_clf_path),
        joblib.load(win_enc_path),
        lgb.Booster(model_file=str(boat1_model_path)),
        joblib.load(boat1_enc_path),
        lgb.Booster(model_file=str(not_boat1_model_path)),
        joblib.load(not_boat1_enc_path),
    )


def prepare_boat1_win_row(race_df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = race_df.copy()
    df["展示順位"] = df["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df["展示"].mean()
    boat1 = df.loc[df["艇"].astype(int) == TARGET_BOAT].copy()
    boat1 = boat1.drop(columns=["日目", "選手名"], errors="ignore")
    boat1, _ = encode_categorical_columns(boat1, encoders, update_encoders=False)
    return boat1[FEATURES]


def prepare_boat1_trifecta(race_df: pd.DataFrame, encoders: dict) -> tuple[pd.DataFrame, np.ndarray]:
    df = race_df.copy()
    df["展示順位"] = df["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df["展示"].mean()
    df = df.loc[df["艇"].astype(int) != TARGET_BOAT].copy()
    boat_numbers = df["艇"].astype(int).values
    df = df.drop(columns=["日目", "選手名"], errors="ignore")
    df, _ = encode_categorical_columns(df, encoders, update_encoders=False)
    return df[FEATURES], boat_numbers


def prepare_not_boat1_trifecta(
    race_df: pd.DataFrame,
    features: list[str],
    encoders: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = race_df.copy()
    df["展示順位"] = df["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df["展示"].mean()
    boat_numbers = df["艇"].astype(int).values
    df = df.drop(columns=["日目", "選手名"], errors="ignore")
    df, _ = encode_categorical_columns(df, encoders, update_encoders=False)
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


def calc_trifecta_probs_from_scores(scores: np.ndarray) -> dict[tuple[int, int, int], float]:
    n = len(scores)
    exp_scores = np.exp(scores - scores.max())
    results: dict[tuple[int, int, int], float] = {}
    for i in range(n):
        p1 = max(exp_scores[i] / exp_scores.sum(), 1e-12)
        rem1 = [b for b in range(n) if b != i]
        sum_rem1 = exp_scores[rem1].sum()
        for j in rem1:
            p2 = max(exp_scores[j] / sum_rem1, 1e-12)
            rem2 = [b for b in rem1 if b != j]
            sum_rem2 = exp_scores[rem2].sum()
            for k in rem2:
                p3 = max(exp_scores[k] / sum_rem2, 1e-12)
                results[(i + 1, j + 1, k + 1)] = p1 * p2 * p3
    return results


def probs_to_ranking(
    prob_map: dict[tuple[int, int, int], float],
) -> list[tuple[tuple[int, int, int], float]]:
    return sorted(prob_map.items(), key=lambda x: x[1], reverse=True)


def get_boat1_trifecta_probs(
    model: lgb.Booster,
    race_df: pd.DataFrame,
    encoders: dict,
) -> dict[tuple[int, int, int], float]:
    x, boat_numbers = prepare_boat1_trifecta(race_df, encoders)
    scores = model.predict(x)
    boat_scores = {int(boat): float(score) for boat, score in zip(boat_numbers, scores)}
    return calc_trifecta_probs_boat1_fixed(boat_scores)


def get_not_boat1_trifecta_probs(
    model: lgb.Booster,
    race_df: pd.DataFrame,
    encoders: dict,
) -> dict[tuple[int, int, int], float]:
    features = model.feature_name()
    x, boat_numbers = prepare_not_boat1_trifecta(race_df, features, encoders)
    scores = model.predict(x)
    score_arr = np.zeros(6)
    for idx, boat in enumerate(boat_numbers):
        score_arr[boat - 1] = scores[idx]
    return calc_trifecta_probs_from_scores(score_arr)


def merge_trifecta_probabilities(
    p_win: float,
    boat1_probs: dict[tuple[int, int, int], float],
    not_boat1_probs: dict[tuple[int, int, int], float],
) -> dict[tuple[int, int, int], float]:
    """
    3モデル出力を1つの3連単確率分布に合成する。

    - boat1_probs: 1着1号艇条件下の PL 確率（合計1）
    - not_boat1_probs: 全6艇 PL 確率（合計1）→ 1着≠1号艇に条件付けして使用
    """
    p_win = float(np.clip(p_win, 0.0, 1.0))
    p_not_win = 1.0 - p_win

    not_boat1_given = {
        combo: prob
        for combo, prob in not_boat1_probs.items()
        if combo[0] != TARGET_BOAT
    }
    norm = sum(not_boat1_given.values())
    if norm <= 0:
        raise ValueError("1着1号艇以外予想モデルの1着≠1号艇確率が0です")

    merged: dict[tuple[int, int, int], float] = {}
    for combo, prob in boat1_probs.items():
        merged[combo] = p_win * prob
    for combo, prob in not_boat1_given.items():
        merged[combo] = merged.get(combo, 0.0) + p_not_win * (prob / norm)
    return merged


def predict_merged_trifecta(
    p_win: float,
    boat1_probs: dict[tuple[int, int, int], float],
    not_boat1_probs: dict[tuple[int, int, int], float],
) -> tuple[tuple[int, int, int], list[tuple[tuple[int, int, int], float]]]:
    ranking = probs_to_ranking(merge_trifecta_probabilities(p_win, boat1_probs, not_boat1_probs))
    return ranking[0][0], ranking


def predict_routed_trifecta(
    pred_win: int,
    boat1_probs: dict[tuple[int, int, int], float],
    not_boat1_probs: dict[tuple[int, int, int], float],
) -> tuple[tuple[int, int, int], list[tuple[tuple[int, int, int], float]]]:
    """参考: 0/1 ハードルーティング"""
    ranking = probs_to_ranking(boat1_probs if pred_win == 1 else not_boat1_probs)
    return ranking[0][0], ranking


def init_trifecta_hits() -> dict[int, int]:
    return {n: 0 for n in TOP_N_LIST}


def update_trifecta_hits(
    hits: dict[int, int],
    actual: tuple[int, int, int],
    ranking: list[tuple[tuple[int, int, int], float]],
) -> None:
    top_combos = [c for c, _ in ranking]
    for n in TOP_N_LIST:
        if actual in top_combos[:n]:
            hits[n] += 1


def evaluate_combined(
    df: pd.DataFrame,
    win_clf: LGBMClassifier,
    win_encoders: dict,
    boat1_model: lgb.Booster,
    boat1_encoders: dict,
    not_boat1_model: lgb.Booster,
    not_boat1_encoders: dict,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    win_preds: list[int] = []
    win_actuals: list[int] = []

    combined_hits = init_trifecta_hits()
    merged_hits = init_trifecta_hits()
    route_stats: dict[str, dict] = {
        ROUTE_BOAT1: {"n": 0, "hits": init_trifecta_hits(), "route_correct": 0},
        ROUTE_NOT_BOAT1: {"n": 0, "hits": init_trifecta_hits(), "route_correct": 0},
    }
    routing_matrix = {
        "pred_win_actual_win": init_trifecta_hits(),
        "pred_win_actual_lose": init_trifecta_hits(),
        "pred_lose_actual_lose": init_trifecta_hits(),
        "pred_lose_actual_win": init_trifecta_hits(),
    }
    routing_matrix_n = {k: 0 for k in routing_matrix}

    n_races = 0
    for key, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue

        actual_win = actual_boat1_win(race_df)
        actual = get_actual_trifecta(race_df)

        x_win = prepare_boat1_win_row(race_df, win_encoders)
        p_win = float(win_clf.predict_proba(x_win)[0, 1])
        pred_win = int(win_clf.predict(x_win)[0])
        win_preds.append(pred_win)
        win_actuals.append(int(actual_win))

        route = ROUTE_BOAT1 if pred_win == 1 else ROUTE_NOT_BOAT1
        route_correct = (pred_win == 1) == actual_win

        boat1_probs = get_boat1_trifecta_probs(boat1_model, race_df, boat1_encoders)
        not_boat1_probs = get_not_boat1_trifecta_probs(
            not_boat1_model, race_df, not_boat1_encoders,
        )
        pred_top, merged_ranking = predict_merged_trifecta(p_win, boat1_probs, not_boat1_probs)
        routed_top, routed_ranking = predict_routed_trifecta(
            pred_win, boat1_probs, not_boat1_probs,
        )

        update_trifecta_hits(merged_hits, actual, merged_ranking)
        update_trifecta_hits(combined_hits, actual, routed_ranking)
        update_trifecta_hits(route_stats[route]["hits"], actual, routed_ranking)
        route_stats[route]["n"] += 1
        if route_correct:
            route_stats[route]["route_correct"] += 1

        if pred_win == 1 and actual_win:
            matrix_key = "pred_win_actual_win"
        elif pred_win == 1 and not actual_win:
            matrix_key = "pred_win_actual_lose"
        elif pred_win == 0 and not actual_win:
            matrix_key = "pred_lose_actual_lose"
        else:
            matrix_key = "pred_lose_actual_win"
        routing_matrix_n[matrix_key] += 1
        update_trifecta_hits(routing_matrix[matrix_key], actual, routed_ranking)

        n_races += 1
        rows.append({
            "開催日": key[0],
            "日目": key[1],
            "レース": key[2],
            "P(1号艇1着)": p_win,
            "1号艇1着予測": pred_win,
            "実際1号艇1着": int(actual_win),
            "ルーティング": route,
            "ルーティング正解": route_correct,
            "実際3連単": "-".join(map(str, actual)),
            "予測3連単": "-".join(map(str, pred_top)),
            "予測確率": merged_ranking[0][1],
            "3連単的中": pred_top == actual,
            "ハードルーティング3連単": "-".join(map(str, routed_top)),
            "ハードルーティング的中": routed_top == actual,
        })

    if n_races == 0:
        raise ValueError("評価対象レースがありません")

    def hit_rates(hits: dict[int, int], n: int) -> dict[str, float]:
        return {f"trifecta_top{k}_rate": hits[k] / n for k in TOP_N_LIST}

    metrics = {
        "n_races": n_races,
        "boat1_win_accuracy": float(accuracy_score(win_actuals, win_preds)),
        "boat1_win_precision": float(precision_score(win_actuals, win_preds, zero_division=0)),
        "boat1_win_recall": float(recall_score(win_actuals, win_preds, zero_division=0)),
        "actual_boat1_win_rate": float(np.mean(win_actuals)),
        "routed_to_boat1_model": route_stats[ROUTE_BOAT1]["n"],
        "routed_to_not_boat1_model": route_stats[ROUTE_NOT_BOAT1]["n"],
        **{f"merged_{k}": v for k, v in hit_rates(merged_hits, n_races).items()},
        **{f"routed_{k}": v for k, v in hit_rates(combined_hits, n_races).items()},
    }

    for route_name, stats in route_stats.items():
        n = stats["n"]
        prefix = "boat1_route" if route_name == ROUTE_BOAT1 else "not_boat1_route"
        metrics[f"{prefix}_n"] = n
        metrics[f"{prefix}_share"] = n / n_races
        metrics[f"{prefix}_route_accuracy"] = stats["route_correct"] / n if n else 0.0
        for k, v in hit_rates(stats["hits"], n).items():
            metrics[f"{prefix}_{k}"] = v if n else 0.0

    matrix_labels = {
        "pred_win_actual_win": "予測1号艇1着×実際1号艇1着",
        "pred_win_actual_lose": "予測1号艇1着×実際1号艇以外1着",
        "pred_lose_actual_lose": "予測1号艇以外1着×実際1号艇以外1着",
        "pred_lose_actual_win": "予測1号艇以外1着×実際1号艇1着",
    }
    for key, label in matrix_labels.items():
        n = routing_matrix_n[key]
        metrics[f"matrix_{key}_n"] = n
        for top_k, rate in hit_rates(routing_matrix[key], n).items():
            metrics[f"matrix_{key}_{top_k}"] = rate if n else 0.0
        metrics[f"matrix_{key}_label"] = label

    return pd.DataFrame(rows), metrics


def print_metrics(metrics: dict) -> None:
    n = metrics["n_races"]
    print(f"\n=== 統合予想 評価（{n} レース） ===")

    print("\n【1号艇勝利予測】")
    print(f"  実際1号艇1着率: {metrics['actual_boat1_win_rate']:.2%}")
    print(f"  Accuracy:       {metrics['boat1_win_accuracy']:.2%}")
    print(f"  Precision:      {metrics['boat1_win_precision']:.2%}")
    print(f"  Recall:         {metrics['boat1_win_recall']:.2%}")
    print(f"  → 1着1号艇予想へ:     {metrics['routed_to_boat1_model']} レース "
          f"({metrics['routed_to_boat1_model'] / n:.1%})")
    print(f"  → 1着1号艇以外予想へ: {metrics['routed_to_not_boat1_model']} レース "
          f"({metrics['routed_to_not_boat1_model'] / n:.1%})")

    print("\n【統合3連単（PL確率合成）】")
    for top_n in TOP_N_LIST:
        print(f"  TOP{top_n}: {metrics[f'merged_trifecta_top{top_n}_rate']:.2%}")

    print("\n【参考: ハードルーティング（0/1振り分け）】")
    for top_n in TOP_N_LIST:
        print(f"  TOP{top_n}: {metrics[f'routed_trifecta_top{top_n}_rate']:.2%}")

    print("\n【ルーティング別 3連単的中率】")
    for prefix, label in (
        ("boat1_route", ROUTE_BOAT1),
        ("not_boat1_route", ROUTE_NOT_BOAT1),
    ):
        rn = metrics[f"{prefix}_n"]
        if rn == 0:
            print(f"  {label}: レースなし")
            continue
        print(f"  {label}: {rn} レース "
              f"(全体の {metrics[f'{prefix}_share']:.1%}, "
              f"ルーティング正解率 {metrics[f'{prefix}_route_accuracy']:.2%})")
        for top_n in TOP_N_LIST:
            print(f"    3連単 TOP{top_n}: {metrics[f'{prefix}_trifecta_top{top_n}_rate']:.2%}")

    print("\n【予測×実際 別 3連単的中率】")
    for key in (
        "pred_win_actual_win", "pred_win_actual_lose",
        "pred_lose_actual_lose", "pred_lose_actual_win",
    ):
        rn = metrics[f"matrix_{key}_n"]
        if rn == 0:
            continue
        print(f"  {metrics[f'matrix_{key}_label']}: {rn} レース")
        for top_n in TOP_N_LIST:
            print(f"    TOP{top_n}: {metrics[f'matrix_{key}_trifecta_top{top_n}_rate']:.2%}")


def save_summary_csv(metrics: dict, path: Path) -> None:
    rows = [
        {"区分": "1号艇勝利予測", "項目": "Accuracy", "値": metrics["boat1_win_accuracy"]},
        {"区分": "1号艇勝利予測", "項目": "Precision", "値": metrics["boat1_win_precision"]},
        {"区分": "1号艇勝利予測", "項目": "Recall", "値": metrics["boat1_win_recall"]},
        {"区分": "ルーティング", "項目": "1着1号艇予想へ", "値": metrics["routed_to_boat1_model"]},
        {"区分": "ルーティング", "項目": "1着1号艇以外予想へ", "値": metrics["routed_to_not_boat1_model"]},
    ]
    for top_n in TOP_N_LIST:
        rows.append({
            "区分": "PL確率合成",
            "項目": f"TOP{top_n}",
            "値": metrics[f"merged_trifecta_top{top_n}_rate"],
        })
        rows.append({
            "区分": "ハードルーティング",
            "項目": f"TOP{top_n}",
            "値": metrics[f"routed_trifecta_top{top_n}_rate"],
        })
    for prefix, label in (("boat1_route", ROUTE_BOAT1), ("not_boat1_route", ROUTE_NOT_BOAT1)):
        rows.append({
            "区分": label,
            "項目": "レース数",
            "値": metrics[f"{prefix}_n"],
        })
        rows.append({
            "区分": label,
            "項目": "ルーティング正解率",
            "値": metrics[f"{prefix}_route_accuracy"],
        })
        for top_n in TOP_N_LIST:
            rows.append({
                "区分": label,
                "項目": f"3連単TOP{top_n}",
                "値": metrics[f"{prefix}_trifecta_top{top_n}_rate"],
            })
    for key in (
        "pred_win_actual_win", "pred_win_actual_lose",
        "pred_lose_actual_lose", "pred_lose_actual_win",
    ):
        label = metrics[f"matrix_{key}_label"]
        rows.append({"区分": label, "項目": "レース数", "値": metrics[f"matrix_{key}_n"]})
        for top_n in TOP_N_LIST:
            rows.append({
                "区分": label,
                "項目": f"TOP{top_n}",
                "値": metrics[f"matrix_{key}_trifecta_top{top_n}_rate"],
            })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="UTF-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3モデル統合予想・評価")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="レースデータCSV（デフォルト: 丸亀テスト用）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("モデル読み込み...")
    win_clf, win_encoders, boat1_model, boat1_encoders, not_boat1_model, not_boat1_encoders = (
        load_models()
    )

    print(f"データ読み込み: {args.data.name}")
    df = pd.read_csv(args.data, low_memory=False)
    configure_features(df)
    df = filter_complete_races(df)
    n_races = df.drop_duplicates(RACE_KEY).shape[0]
    print(f"  対象: {len(df)} 行 / {n_races} レース")
    print(
        f"  期間: {pd.to_datetime(df['開催日']).min().date()} 〜 "
        f"{pd.to_datetime(df['開催日']).max().date()}"
    )

    predictions, metrics = evaluate_combined(
        df, win_clf, win_encoders,
        boat1_model, boat1_encoders,
        not_boat1_model, not_boat1_encoders,
    )
    print_metrics(metrics)

    pred_path = OUTPUT_DIR / "予測結果.csv"
    eval_json_path = OUTPUT_DIR / "評価結果.json"
    eval_csv_path = OUTPUT_DIR / "評価結果.csv"

    predictions.to_csv(pred_path, index=False, encoding="UTF-8-sig")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_summary_csv(metrics, eval_csv_path)

    print(f"\n保存完了:")
    print(f"  {pred_path}")
    print(f"  {eval_json_path}")
    print(f"  {eval_csv_path}")


if __name__ == "__main__":
    main()
