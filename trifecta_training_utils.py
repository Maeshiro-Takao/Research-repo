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


def _resolve_base_value(explainer: shap.TreeExplainer):
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        return float(np.asarray(expected).reshape(-1)[0])
    return float(expected)


def run_shap_analysis(
    model: lgb.Booster,
    x_sample: pd.DataFrame,
    beeswarm_path: Path,
    waterfall_path: Path,
    title: str = "",
) -> None:
    """
    SHAP分析を出力する。

    - Summary Plot (Beeswarm): 日本語特徴量名
    - Waterfall Plot: 代表サンプル1件、日本語特徴量名
    """
    x_sample = x_sample.copy()
    feature_names = [str(c) for c in x_sample.columns]
    x_sample.columns = feature_names

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
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
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=_resolve_base_value(explainer),
        data=x_sample.iloc[sample_idx].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.gcf().savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  SHAP Beeswarm: {beeswarm_path}")
    print(f"  SHAP Waterfall: {waterfall_path}")
