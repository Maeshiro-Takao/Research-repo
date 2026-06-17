"""3連単学習スクリプト共通: 時系列重み付け・年次学習"""
from __future__ import annotations

from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

# 新しいレースほど重みを大きく（半減期1年）
RECENCY_HALF_LIFE_YEARS = 1.0
# Optuna の検証に使う直近1年
VALIDATION_YEARS = 1

# 直近10年/5年の勝率・連帯率（重み付けで代替するため除外）
ROLLING_FEATURE_EXCLUDE = {
    "直近10年当地勝率", "直近10年2連帯率", "直近10年3連帯率",
    "直近5年当地勝率", "直近5年2連帯率", "直近5年3連帯率",
    "直近10年勝率", "直近5年勝率",
}

LGBM_PARAM_KEYS = (
    "num_leaves", "max_depth", "learning_rate", "min_data_in_leaf",
    "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2",
)


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
        "years": years,
        "rounds_per_year": rounds_per_year,
        "yearly_log": yearly_log,
    }
    return model, encoders, x_train, meta
