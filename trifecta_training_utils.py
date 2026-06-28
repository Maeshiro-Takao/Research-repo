"""3連単学習スクリプト共通: 特徴量・時系列重み・年次学習・SHAP"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import shap

# 新しいレースほど重みを大きく（半減期1年）
RECENCY_HALF_LIFE_YEARS = 1.0
# Optuna の検証に使う直近1年
VALIDATION_YEARS = 1

RACE_EXCLUDE_COLUMNS = {
    "レース", "着", "選手名", "日目", "開催日", "登番", "モーター", "ボート", "艇", "3連単オッズ",
}
PLAYER_META_COLS = ["算出期間自", "算出期間至"]
RACE_BASE_FEATURES = [
    "展示",
    "展示順位", "展示差", "風速", "波高", "天気", "風向",
    "当地勝率",
]
CATEGORICAL_FEATURES = ["天気", "風向", "級"]
UNKNOWN_CATEGORY_TOKEN = "__UNKNOWN__"

# 方式比較評価で使用する TOP N（現在方式の evaluate_trifecta とは別）
TRIFECTA_COMPARE_TOP_N_LIST = [1, 3, 5, 10, 15, 20]
SCORING_METHOD_PROB = "現在方式"
SCORING_METHOD_RANK = "順位スコア方式"

LGBM_PARAM_KEYS = (
    "num_leaves", "max_depth", "learning_rate", "min_data_in_leaf",
    "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2",
)

BASE_LGBM_RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "verbosity": -1,
    "seed": 42,
    "feature_pre_filter": False,
}

JAPANESE_FONT_CANDIDATES = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Hiragino Maru Gothic ProN",
    "Yu Gothic",
    "YuGothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAGothic",
    "MS Gothic",
]


def build_feature_list(df: pd.DataFrame) -> list[str]:
    """学習に使用する日本語特徴量名リストを構築"""
    base = [c for c in RACE_BASE_FEATURES if c in df.columns]
    exclude = (
        RACE_EXCLUDE_COLUMNS
        | set(PLAYER_META_COLS)
        | set(RACE_BASE_FEATURES)
        | {"登番"}
    )
    player_features = [c for c in df.columns if c not in exclude]
    return list(dict.fromkeys(base + player_features))


def encode_categorical_columns(
    df: pd.DataFrame,
    encoders: dict | None,
    categorical_features: list[str] | None = None,
    *,
    update_encoders: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    カテゴリ列を LightGBM 向けの非負整数にエンコードする。

    - update_encoders=True: 新カテゴリを辞書へ追加（年次增量学習向け）
    - update_encoders=False: 未登録値は __UNKNOWN__ へ（推論向け）
    """
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    encoders = {} if encoders is None else encoders
    df = df.copy()

    for col in categorical_features:
        if col not in df.columns:
            continue
        if col not in encoders:
            encoders[col] = {}

        values = df[col].astype(str)
        if update_encoders:
            next_idx = max(encoders[col].values(), default=-1) + 1
            for value in values.unique():
                if value not in encoders[col]:
                    encoders[col][value] = next_idx
                    next_idx += 1

        mapped = values.map(encoders[col])
        if mapped.isna().any():
            if UNKNOWN_CATEGORY_TOKEN not in encoders[col]:
                unknown_idx = max(encoders[col].values(), default=-1) + 1
                encoders[col][UNKNOWN_CATEGORY_TOKEN] = unknown_idx
            mapped = mapped.fillna(encoders[col][UNKNOWN_CATEGORY_TOKEN])

        df[col] = mapped.astype(int)

    return df, encoders


def setup_japanese_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in JAPANESE_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    for font in font_manager.fontManager.ttflist:
        if any(k in font.name for k in ("Hiragino", "Noto Sans CJK", "Yu Gothic", "Meiryo")):
            plt.rcParams["font.family"] = font.name
            plt.rcParams["axes.unicode_minus"] = False
            return font.name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def compute_recency_weights(dates: pd.Series) -> np.ndarray:
    """開催日が新しいほど重みを大きくする（平均1に正規化）"""
    dt = pd.to_datetime(dates)
    years_ago = (dt.max() - dt).dt.days / 365.25
    weights = np.power(0.5, years_ago / RECENCY_HALF_LIFE_YEARS)
    return (weights / weights.mean()).to_numpy()


def temporal_train_valid_split(
    df: pd.DataFrame,
    valid_years: int = VALIDATION_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """直近 valid_years 年を検証、それ以前を学習用に分割"""
    dates = pd.to_datetime(df["開催日"])
    cutoff = dates.max() - pd.DateOffset(years=valid_years)
    train_df = df.loc[dates <= cutoff].copy()
    valid_df = df.loc[dates > cutoff].copy()
    return train_df, valid_df


def get_calendar_years(df: pd.DataFrame) -> list[int]:
    return sorted(pd.to_datetime(df["開催日"]).dt.year.unique().tolist())


def build_lgbm_params(best_params: dict, base_params: dict) -> dict:
    params = {**base_params}
    for key in LGBM_PARAM_KEYS:
        params[key] = best_params[key]
    return params


def get_actual_trifecta(race_df: pd.DataFrame) -> tuple[int, int, int]:
    top3 = race_df.sort_values("着").head(3)
    return tuple(top3["艇"].astype(int).tolist())  # type: ignore[return-value]


def calc_trifecta_probs_boat1_fixed(
    boat_scores: dict[int, float],
    first_boat: int = 1,
) -> dict[tuple[int, int, int], float]:
    """1着固定モデル: exp() による3連単確率（現在方式）"""
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
            results[(first_boat, second, third)] = p2 * p3
    return results


def calc_trifecta_probs_from_scores(scores: np.ndarray) -> dict[tuple[int, int, int], float]:
    """全6艇モデル: Plackett–Luce による3連単確率（後方互換ラッパー）"""
    from plackett_luce import plackett_luce_trifecta_probs
    return plackett_luce_trifecta_probs(scores)


def build_rank_trifecta_boat1_fixed(
    boat_scores: dict[int, float],
    first_boat: int = 1,
) -> list[tuple[int, int, int]]:
    """1着固定モデル: スコア順位のみで3連単候補を生成（順位スコア方式）"""
    ranked = sorted(boat_scores.keys(), key=lambda b: (-boat_scores[b], b))
    return [
        (first_boat, second, third)
        for i, second in enumerate(ranked)
        for third in ranked[i + 1 :]
    ]


def build_rank_trifecta_all_boats(
    boat_scores: dict[int, float],
) -> list[tuple[int, int, int]]:
    """全6艇モデル: スコア順位のみで3連単候補を生成（順位スコア方式）"""
    ranked = sorted(boat_scores.keys(), key=lambda b: (-boat_scores[b], b))
    result: list[tuple[int, int, int]] = []
    for i, first in enumerate(ranked):
        for j in range(i + 1, len(ranked)):
            second = ranked[j]
            for third in ranked[j + 1 :]:
                result.append((first, second, third))
    return result


def trifecta_ranking_prob_boat1_fixed(
    boat_scores: dict[int, float],
    first_boat: int = 1,
) -> list[tuple[int, int, int]]:
    prob_map = calc_trifecta_probs_boat1_fixed(boat_scores, first_boat=first_boat)
    return [combo for combo, _ in sorted(prob_map.items(), key=lambda x: x[1], reverse=True)]


def trifecta_ranking_prob_all_boats(boat_scores: dict[int, float]) -> list[tuple[int, int, int]]:
    score_arr = np.array([boat_scores.get(b, 0.0) for b in range(1, 7)])
    prob_map = calc_trifecta_probs_from_scores(score_arr)
    return [combo for combo, _ in sorted(prob_map.items(), key=lambda x: x[1], reverse=True)]


def _aggregate_trifecta_hits(
    ranking: list[tuple[int, int, int]],
    actual: tuple[int, int, int],
    top_n_list: list[int],
) -> dict[str, int]:
    pred_top = ranking[0]
    hits = {f"top{n}": int(actual in ranking[:n]) for n in top_n_list}
    return {
        "first_hit": int(pred_top[0] == actual[0]),
        "exacta_hit": int(pred_top[:2] == actual[:2]),
        **hits,
    }


def evaluate_dual_scoring_methods(
    df: pd.DataFrame,
    race_key: list[str],
    label: str,
    get_boat_scores: Callable[[pd.DataFrame], dict[int, float]],
    boat1_fixed: bool,
    top_n_list: list[int] | None = None,
) -> tuple[dict, dict]:
    """
    現在方式（exp確率）と順位スコア方式を同一データで評価する。
    """
    top_n_list = top_n_list or TRIFECTA_COMPARE_TOP_N_LIST
    prob_totals = {"first_hit": 0, "exacta_hit": 0, **{f"top{n}": 0 for n in top_n_list}}
    rank_totals = {"first_hit": 0, "exacta_hit": 0, **{f"top{n}": 0 for n in top_n_list}}
    n_races = 0

    prob_builder = trifecta_ranking_prob_boat1_fixed if boat1_fixed else trifecta_ranking_prob_all_boats
    rank_builder = build_rank_trifecta_boat1_fixed if boat1_fixed else build_rank_trifecta_all_boats

    for _, race_df in df.groupby(race_key, sort=False):
        if len(race_df) != 6:
            continue
        actual = get_actual_trifecta(race_df)
        boat_scores = get_boat_scores(race_df)
        prob_ranking = prob_builder(boat_scores)
        rank_ranking = rank_builder(boat_scores)

        prob_hits = _aggregate_trifecta_hits(prob_ranking, actual, top_n_list)
        rank_hits = _aggregate_trifecta_hits(rank_ranking, actual, top_n_list)
        for key in prob_totals:
            prob_totals[key] += prob_hits[key]
            rank_totals[key] += rank_hits[key]
        n_races += 1

    if n_races == 0:
        raise ValueError(f"評価対象レースがありません: {label}")

    def to_metrics(totals: dict, method: str) -> dict:
        metrics = {
            "label": label,
            "method": method,
            "n_races": n_races,
            "first_hit_rate": totals["first_hit"] / n_races,
            "exacta_hit_rate": totals["exacta_hit"] / n_races,
        }
        for n in top_n_list:
            metrics[f"trifecta_top{n}_rate"] = totals[f"top{n}"] / n_races
        return metrics

    return to_metrics(prob_totals, SCORING_METHOD_PROB), to_metrics(rank_totals, SCORING_METHOD_RANK)


def build_scoring_method_comparison_table(
    prob_metrics: dict,
    rank_metrics: dict,
    top_n_list: list[int] | None = None,
) -> pd.DataFrame:
    top_n_list = top_n_list or TRIFECTA_COMPARE_TOP_N_LIST
    rows = [
        {
            "評価項目": "1着艇番的中率",
            SCORING_METHOD_PROB: prob_metrics["first_hit_rate"],
            SCORING_METHOD_RANK: rank_metrics["first_hit_rate"],
        },
        {
            "評価項目": "2連単的中率",
            SCORING_METHOD_PROB: prob_metrics["exacta_hit_rate"],
            SCORING_METHOD_RANK: rank_metrics["exacta_hit_rate"],
        },
    ]
    for n in top_n_list:
        rows.append({
            "評価項目": f"3連単TOP{n}",
            SCORING_METHOD_PROB: prob_metrics[f"trifecta_top{n}_rate"],
            SCORING_METHOD_RANK: rank_metrics[f"trifecta_top{n}_rate"],
        })
    return pd.DataFrame(rows)


def print_scoring_method_comparison(
    model_name: str,
    prob_metrics: dict,
    rank_metrics: dict,
    top_n_list: list[int] | None = None,
) -> None:
    table = build_scoring_method_comparison_table(prob_metrics, rank_metrics, top_n_list)
    print(f"\n【{model_name}】方式比較 ({prob_metrics['label']})")
    print(f"  レース数: {prob_metrics['n_races']}")
    print(f"  {'評価項目':<16} | {SCORING_METHOD_PROB:>12} | {SCORING_METHOD_RANK:>14}")
    print(f"  {'-' * 16}-+-{'-' * 12}-+-{'-' * 14}")
    for _, row in table.iterrows():
        prob_val = f"{row[SCORING_METHOD_PROB]:.2%}"
        rank_val = f"{row[SCORING_METHOD_RANK]:.2%}"
        print(f"  {row['評価項目']:<16} | {prob_val:>12} | {rank_val:>14}")


def save_scoring_method_comparison(
    output_path: Path,
    model_name: str,
    prob_metrics: dict,
    rank_metrics: dict,
    top_n_list: list[int] | None = None,
) -> pd.DataFrame:
    table = build_scoring_method_comparison_table(prob_metrics, rank_metrics, top_n_list)
    table.insert(0, "モデル", model_name)
    table.insert(1, "データ", prob_metrics["label"])
    table.insert(2, "レース数", prob_metrics["n_races"])
    table.to_csv(output_path, index=False, encoding="UTF-8-sig")
    print(f"  方式比較CSV: {output_path}")
    return table


def train_incremental_by_year(
    df: pd.DataFrame,
    best_params: dict,
    base_params: dict,
    prepare: Callable,
    build_rank_dataset: Callable,
) -> tuple[lgb.Booster, dict, pd.DataFrame, dict]:
    """カレンダー年ごとに累積データで增量学習（init_model で継続）"""
    years = get_calendar_years(df)
    if not years:
        raise ValueError("学習データに有効な年がありません")

    params = build_lgbm_params(best_params, base_params)
    rounds_per_year = max(30, best_params["num_boost_round"] // len(years))
    encoders: dict = {}
    model: lgb.Booster | None = None
    x_train = None
    dates = pd.to_datetime(df["開催日"])
    yearly_log: list[dict] = []

    print(
        f"  年次增量学習: {years[0]}〜{years[-1]}年 "
        f"({len(years)}ステップ, 各{rounds_per_year}round)"
    )

    for year in years:
        cumulative = df.loc[dates.dt.year <= year].copy()
        n_races = cumulative.drop_duplicates(["開催日", "日目", "レース"]).shape[0]
        if n_races < 30:
            continue

        _, _, _, encoders, _, _ = prepare(cumulative, encoders)
        train_set, x = build_rank_dataset(cumulative, encoders)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=rounds_per_year,
            init_model=model,
            callbacks=[lgb.log_evaluation(0)],
        )
        x_train = x
        yearly_log.append({"year": year, "cumulative_races": n_races})
        print(f"    {year}年: 累計 {n_races} レース")

    if model is None or x_train is None:
        raise ValueError("年次增量学習でモデルを構築できませんでした")

    meta = {
        "training_mode": "yearly_incremental",
        "algorithm": "LightGBM Ranker (lambdarank)",
        "years": years,
        "rounds_per_year": rounds_per_year,
        "yearly_log": yearly_log,
    }
    return model, encoders, x_train, meta


def _resolve_base_value(explainer: shap.TreeExplainer, class_idx: int | None = None) -> float:
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        values = np.asarray(expected).reshape(-1)
        if class_idx is not None:
            return float(values[class_idx])
        return float(values[0])
    return float(expected)


def _predicted_classes(model: lgb.Booster, x_sample: pd.DataFrame) -> np.ndarray | None:
    preds = np.asarray(model.predict(x_sample))
    if preds.ndim == 2 and preds.shape[1] > 1:
        return np.argmax(preds, axis=1)
    return None


def _shap_row_for_waterfall(shap_values, sample_idx: int, class_idx: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx][sample_idx])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[sample_idx, :, class_idx]
    return arr[sample_idx]


def _shap_values_for_summary(shap_values):
    if isinstance(shap_values, list):
        return shap_values
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return [arr[:, :, c] for c in range(arr.shape[2])]
    return arr


def run_shap_analysis(
    model: lgb.Booster,
    x_sample: pd.DataFrame,
    beeswarm_path: Path,
    waterfall_path: Path,
    title: str = "",
) -> None:
    """
    SHAP分析を出力する。

    - Summary Plot (Beeswarm): 日本語特徴量名（多クラス対応）
    - Waterfall Plot: 代表サンプル1件（予測クラスの SHAP）
    """
    x_sample = x_sample.copy()
    feature_names = [str(c) for c in x_sample.columns]
    x_sample.columns = feature_names

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    pred_classes = _predicted_classes(model, x_sample)
    plot_values = _shap_values_for_summary(shap_values)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        plot_values,
        x_sample,
        feature_names=feature_names,
        show=False,
    )
    if title:
        plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    sample_idx = min(len(x_sample) // 2, len(x_sample) - 1)
    class_idx = int(pred_classes[sample_idx]) if pred_classes is not None else 0
    explanation = shap.Explanation(
        values=_shap_row_for_waterfall(shap_values, sample_idx, class_idx),
        base_values=_resolve_base_value(explainer, class_idx if pred_classes is not None else None),
        data=x_sample.iloc[sample_idx].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.gcf().savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  SHAP Beeswarm: {beeswarm_path}")
    print(f"  SHAP Waterfall: {waterfall_path}")
