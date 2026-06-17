"""
抽出データから派生項目を計算する。

- 当地勝率・直近10年/5年当地勝率
- 直近10年/5年の2連帯率・3連帯率（全国成績）
- 1着率・2着率・3着率

ComRD から import して使用する。
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import pandas as pd

from ExtRD import attach_grade_info, extract_player_records

RACE_ROW_KEY = ["開催日", "日目", "レース", "艇"]

PLAYER_OUTPUT_COLUMNS = [
    "登番", "名前漢字", "級", "身長", "体重", "勝率", "複勝率",
    "直近10年2連帯率", "直近10年3連帯率",
    "直近5年2連帯率", "直近5年3連帯率",
    "1着率", "2着率", "3着率",
    "優出回数", "優勝回数", "平均スタートタイミング",
]
for _course in range(1, 7):
    PLAYER_OUTPUT_COLUMNS += [
        f"{_course}コース進入回数",
        f"{_course}コース複勝率",
        f"{_course}コース平均スタートタイミング",
        f"{_course}コース平均スタート順位",
    ]
PLAYER_OUTPUT_COLUMNS += ["算出期間自", "算出期間至"]

PLAYER_MERGE_COLS = [
    c for c in PLAYER_OUTPUT_COLUMNS if c not in {"登番", "名前漢字"}
]

LOCAL_WIN_RATE_COLUMNS = ["当地勝率", "直近10年当地勝率", "直近5年当地勝率"]

PLACE_POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
LOCAL_RATE_META_COLS = [
    "開催日", "日目", "レース", "艇", "登番", "着", "grade", "is_championship",
]

ROLLING_WINDOWS = (10, 5)
RANK3_COLUMNS = [f"{c}コース3着回数" for c in range(1, 7)]
ROLLING_RATE_COLUMNS = [
    "直近10年2連帯率", "直近10年3連帯率",
    "直近5年2連帯率", "直近5年3連帯率",
]
ROLLING_LOCAL_WINDOWS = (10, 5)
DEDUP_KEY = ["登番", "年", "期"]


def round_rate(value: float | None, decimals: int) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), decimals)


def add_third_place_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_3着回数"] = df[RANK3_COLUMNS].fillna(0).sum(axis=1).astype(int)
    return df


def compute_rolling_rates(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """算出期間至を基準に直近10年/5年の2連帯率・3連帯率を付与"""
    target = add_third_place_count(df)
    if history_df is not None:
        pool = pd.concat([add_third_place_count(history_df), target], ignore_index=True)
    else:
        pool = target.copy()
    pool = pool.drop_duplicates(subset=DEDUP_KEY, keep="last")
    pool["算出期間至"] = pd.to_datetime(pool["算出期間至"])
    pool = pool.sort_values(["登番", "算出期間至"]).reset_index(drop=True)

    rate_values: dict[tuple, dict[str, float | None]] = {}
    for _, group in pool.groupby("登番", sort=False):
        group = group.reset_index(drop=True)
        for _, row in group.iterrows():
            ref_date = row["算出期間至"]
            key = (row["登番"], row["年"], row["期"])
            rates: dict[str, float | None] = {}

            for years in ROLLING_WINDOWS:
                start_date = ref_date - pd.DateOffset(years=years)
                hist = group[
                    (group["算出期間至"] <= ref_date)
                    & (group["算出期間至"] > start_date)
                ]
                total_runs = hist["出走回数"].fillna(0).sum()
                prefix = f"直近{years}年"

                if total_runs <= 0:
                    rates[f"{prefix}2連帯率"] = None
                    rates[f"{prefix}3連帯率"] = None
                    continue

                first = hist["1着回数"].fillna(0).sum()
                second = hist["2着回数"].fillna(0).sum()
                third = hist["_3着回数"].fillna(0).sum()

                rates[f"{prefix}2連帯率"] = round_rate((first + second) / total_runs * 100, 1)
                rates[f"{prefix}3連帯率"] = round_rate((first + second + third) / total_runs * 100, 1)

            rate_values[key] = rates

    for col in ROLLING_RATE_COLUMNS:
        target[col] = [
            rate_values.get((row["登番"], row["年"], row["期"]), {}).get(col)
            for _, row in target.iterrows()
        ]

    return target.drop(columns=["_3着回数"])


def add_place_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    third = df[RANK3_COLUMNS].fillna(0).sum(axis=1)
    runs = df["出走回数"].fillna(0)

    for name, counts in (
        ("1着率", df["1着回数"].fillna(0)),
        ("2着率", df["2着回数"].fillna(0)),
        ("3着率", third),
    ):
        df[name] = [
            round_rate(c / run * 100, 1) if run > 0 else None
            for c, run in zip(counts, runs)
        ]
    return df


def build_player_dataframe(
    records: list[dict],
    history_records: list[dict] | None = None,
) -> pd.DataFrame:
    """抽出済み選手データに計算列を付与"""
    df = pd.DataFrame(records)
    history_df = pd.DataFrame(history_records) if history_records else None
    df = compute_rolling_rates(df, history_df)
    df = add_place_rates(df)
    return df[PLAYER_OUTPUT_COLUMNS]


def _rolling_local_rate(
    dates: list[int],
    prefix_sums: list[int],
    ref_ts: int,
    years: int,
) -> float | None:
    if not dates:
        return None
    start_ts = ref_ts - years * 365.25 * 24 * 3600
    start_idx = bisect.bisect_right(dates, int(start_ts))
    end_idx = bisect.bisect_right(dates, ref_ts)
    count = end_idx - start_idx
    if count <= 0:
        return None
    total_points = prefix_sums[end_idx] - prefix_sums[start_idx]
    return round_rate(10 * np.sqrt(total_points / count / 10), 2)


def grade_point_bonus(grade: str | None) -> int:
    if grade == "SG":
        return 2
    if grade in {"G1", "G2"}:
        return 1
    return 0


def championship_point_bonus(is_championship: bool) -> int:
    return 1 if is_championship else 0


def calc_place_points(rank: int, grade: str | None, is_championship: bool) -> int:
    base = PLACE_POINTS.get(rank, 0)
    return (
        base
        + grade_point_bonus(grade)
        + championship_point_bonus(is_championship)
    )


def _calc_win_rate(stats: dict | None) -> float | None:
    if not stats or stats["runs"] <= 0:
        return None
    return round_rate(10 * np.sqrt(stats["points"] / stats["runs"] / 10), 2)


def _update_player_stats(stats: dict, points: int) -> None:
    stats["runs"] += 1
    stats["points"] += points


def _empty_player_stats() -> dict:
    return {"runs": 0, "points": 0}


def _as_bool(value) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def add_local_win_rate(
    race_df: pd.DataFrame,
    grade_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    丸亀グレード別レースCSVのグレード・優勝戦を使い、当地勝率を算出する。

    着順点: 基本点 + グレード加算(SG+2/G1・G2+1) + 優勝戦加算(+1)
    当地勝率 = 10 × √(平均着順点 ÷ 10)（当レース以前の丸亀成績）
    """
    work_df = attach_grade_info(race_df, grade_df)
    target_rates = work_df[LOCAL_RATE_META_COLS].copy()

    if history_df is not None and len(history_df) > 0:
        timeline = pd.concat([history_df[LOCAL_RATE_META_COLS], target_rates], ignore_index=True)
        target_len = len(target_rates)
    else:
        timeline = target_rates.copy()
        target_len = len(work_df)

    timeline = timeline.sort_values(
        ["開催日", "日目", "レース", "艇"]
    ).reset_index(drop=True)

    player_stats: dict[int, dict] = {}
    player_dates: dict[int, list[int]] = {}
    player_prefix: dict[int, list[int]] = {}
    local_rates: list[float | None] = []
    roll_10_local: list[float | None] = []
    roll_5_local: list[float | None] = []

    for row in timeline.itertuples(index=False):
        toban = int(row.登番)
        race_ts = int(pd.Timestamp(row.開催日).timestamp())
        rank = int(row.着)
        grade = str(row.grade) if pd.notna(row.grade) else "一般"
        is_championship = _as_bool(row.is_championship)
        stats = player_stats.setdefault(toban, _empty_player_stats())
        dates = player_dates.setdefault(toban, [])
        prefix = player_prefix.setdefault(toban, [0])

        local_rates.append(_calc_win_rate(stats))
        roll_10_local.append(_rolling_local_rate(dates, prefix, race_ts, 10))
        roll_5_local.append(_rolling_local_rate(dates, prefix, race_ts, 5))

        points = calc_place_points(rank, grade, is_championship)
        _update_player_stats(stats, points)
        dates.append(race_ts)
        prefix.append(prefix[-1] + points)

    result = work_df.copy()
    result["当地勝率"] = local_rates[-target_len:]
    result["直近10年当地勝率"] = roll_10_local[-target_len:]
    result["直近5年当地勝率"] = roll_5_local[-target_len:]
    rate_meta = result[LOCAL_RATE_META_COLS].copy()
    return result.drop(columns=["grade", "is_championship"]), rate_meta


def _match_player_stats(race: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """レース行へ選手統計を付与（期間一致、未一致時は直前の期を使用）"""
    player_cols = [c for c in player_df.columns if c in PLAYER_MERGE_COLS]
    player_groups = {
        int(toban): group.sort_values("算出期間自").reset_index(drop=True)
        for toban, group in player_df.groupby("登番")
    }

    matched_cols = {col: [] for col in player_cols}
    for row in race.itertuples(index=False):
        toban = int(row.登番)
        race_date = row.開催日
        group = player_groups.get(toban)
        if group is None or len(group) == 0:
            for col in player_cols:
                matched_cols[col].append(pd.NA)
            continue

        exact = group[
            (group["算出期間自"] <= race_date) & (race_date <= group["算出期間至"])
        ]
        if len(exact) > 0:
            pick = exact.iloc[-1]
        else:
            prior = group[group["算出期間至"] <= race_date]
            if len(prior) == 0:
                for col in player_cols:
                    matched_cols[col].append(pd.NA)
                continue
            pick = prior.iloc[-1]

        for col in player_cols:
            matched_cols[col].append(pick[col])

    out = race.copy()
    for col in player_cols:
        out[col] = matched_cols[col]
    return out


def attach_player_stats(
    race_df: pd.DataFrame,
    years: range,
    player_history: list[dict] | None,
    player_years: range | None = None,
) -> pd.DataFrame:
    """計算済み選手統計をレース行へ付与

    fan*.txt は算出期間の終了月ごとに翌年フォルダへ置かれるため、
    レース年の翌年分も player_years に含めて期間マッチさせる。
    """
    if player_years is None:
        player_years = range(years.start, years.stop + 1)

    player_df = build_player_dataframe(
        extract_player_records(player_years),
        player_history,
    )

    race = race_df.copy()
    race["開催日"] = pd.to_datetime(race["開催日"])
    player_df["算出期間自"] = pd.to_datetime(player_df["算出期間自"])
    player_df["算出期間至"] = pd.to_datetime(player_df["算出期間至"])

    out = _match_player_stats(race, player_df)
    out["開催日"] = out["開催日"].dt.strftime("%Y-%m-%d")
    return out


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "編集データ"

DATASETS = [
    {"years": range(14, 25), "output": "丸亀学習用_選手データ.csv"},
    {"years": range(25, 26), "output": "丸亀テスト用_選手データ.csv"},
]


def main():
    """計算済み選手データをCSV出力（デバッグ・確認用）"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history_records: list[dict] | None = None
    for dataset in DATASETS:
        print(f"\n=== {dataset['output']} ===")
        records = extract_player_records(dataset["years"])
        df = build_player_dataframe(records, history_records)

        output_path = OUTPUT_DIR / dataset["output"]
        df.to_csv(output_path, index=False, encoding="UTF-8-sig")

        print("総件数:", len(df))
        if len(df) > 0:
            meta = pd.DataFrame(records)
            print("登番数:", df["登番"].nunique())
            print("年×期:", meta.groupby(["年", "期"]).size().to_dict())
            print("重複件数:", meta.duplicated(DEDUP_KEY).sum())
            print("保存:", output_path)

        if dataset["output"] == "丸亀学習用_選手データ.csv":
            history_records = records


if __name__ == "__main__":
    main()
