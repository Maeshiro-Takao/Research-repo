"""
ExtRD.py で抽出した DataFrame から学習用特徴量を計算する。

- txt から直接取得できない項目は全てここで計算
- 計算結果のみを DataFrame として返す

ComRD から import して使用する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ExtRD import RACE_ROW_KEY, pick_player_period_row

RACE_KEY = ["開催日", "日目", "レース"]

PLAYER_COMPUTED_COLS = ["1着率", "2着率", "3着率"]

LOCAL_COMPUTED_COLS = ["当地勝率"]

COMPUTED_COLUMNS = LOCAL_COMPUTED_COLS + PLAYER_COMPUTED_COLS

PLACE_POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
LOCAL_RATE_META_COLS = [
    "開催日", "日目", "レース", "艇", "登番", "着", "grade", "is_championship",
]

RANK3_COLUMNS = [f"{c}コース3着回数" for c in range(1, 7)]
DEDUP_KEY = ["登番", "年", "期"]

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "編集データ"


def round_rate(value: float | None, decimals: int) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), decimals)


def attach_grade_info(
    race_df: pd.DataFrame,
    grade_df: pd.DataFrame,
) -> pd.DataFrame:
    """丸亀グレード別レースCSVからグレード・優勝戦をレース行へ付与"""
    grade_cols = grade_df.rename(
        columns={"グレード": "grade", "優勝戦": "is_championship"}
    )
    merged = race_df.merge(
        grade_cols[["開催日", "日目", "レース", "grade", "is_championship"]],
        on=["開催日", "日目", "レース"],
        how="left",
    )
    merged["grade"] = merged["grade"].map(normalize_grade)
    merged["is_championship"] = merged["is_championship"].fillna(False).astype(bool)
    return merged


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


def build_player_computed_df(records: list[dict]) -> pd.DataFrame:
    """選手生データから計算列のみを生成"""
    df = pd.DataFrame(records)
    if len(df) == 0:
        return pd.DataFrame(columns=["登番", "算出期間自", "算出期間至"] + PLAYER_COMPUTED_COLS)

    df = add_place_rates(df)
    df["算出期間自"] = pd.to_datetime(df["算出期間自"])
    df["算出期間至"] = pd.to_datetime(df["算出期間至"])
    return df[["登番", "算出期間自", "算出期間至"] + PLAYER_COMPUTED_COLS]


def attach_computed_player_fields(
    race_df: pd.DataFrame,
    player_computed_df: pd.DataFrame,
) -> pd.DataFrame:
    """計算済み選手統計をレース行キーへ付与"""
    player_groups = {
        int(toban): group.sort_values("算出期間自").reset_index(drop=True)
        for toban, group in player_computed_df.groupby("登番")
    }

    matched_cols = {col: [] for col in PLAYER_COMPUTED_COLS}
    race = race_df.copy()
    race["開催日"] = pd.to_datetime(race["開催日"])

    for row in race.itertuples(index=False):
        group = player_groups.get(int(row.登番))
        if group is None or len(group) == 0:
            for col in PLAYER_COMPUTED_COLS:
                matched_cols[col].append(pd.NA)
            continue

        pick = pick_player_period_row(group, row.開催日)
        if pick is None:
            for col in PLAYER_COMPUTED_COLS:
                matched_cols[col].append(pd.NA)
            continue

        for col in PLAYER_COMPUTED_COLS:
            matched_cols[col].append(pick[col])

    out = race_df[RACE_ROW_KEY].copy()
    for col in PLAYER_COMPUTED_COLS:
        out[col] = matched_cols[col]
    return out


def normalize_grade(grade: str | None) -> str:
    if grade is None or pd.isna(grade):
        return "一般"
    text = str(grade).strip().upper()
    if text in {"SG", "G1", "G2"}:
        return text
    return "一般"


def grade_point_bonus(grade: str | None) -> int:
    """SG+2点、G1/G2+1点、一般+0点"""
    match normalize_grade(grade):
        case "SG":
            return 2
        case "G1" | "G2":
            return 1
        case _:
            return 0


def championship_point_bonus(is_championship: bool) -> int:
    """各グレード（一般/SG/G1/G2）の優勝戦は+1点"""
    return 1 if is_championship else 0


def calc_place_points(rank: int, grade: str | None, is_championship: bool) -> int:
    """
    着順点 = 基本点（一般レース含む全レース） + グレード加算 + 優勝戦加算

    - 基本点: 1着10 / 2着8 / 3着6 / 4着4 / 5着2 / 6着1
    - SG競走: +2点、G1/G2競走: +1点
    - 各競走の優勝戦: さらに+1点
    """
    if rank not in PLACE_POINTS:
        return 0
    base = PLACE_POINTS[rank]
    return (
        base
        + grade_point_bonus(grade)
        + championship_point_bonus(is_championship)
    )


def _calc_win_rate(stats: dict | None) -> float:
    """当地勝率 = 10 × √(平均着順点 ÷ 10)。算出不可は 0"""
    if not stats or stats["runs"] <= 0:
        return 0.0
    value = 10 * np.sqrt(stats["points"] / stats["runs"] / 10)
    return round(float(value), 2)


def _update_player_stats(stats: dict, points: int) -> None:
    stats["runs"] += 1
    stats["points"] += points


def _empty_player_stats() -> dict:
    return {"runs": 0, "points": 0}


def _as_bool(value) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def compute_local_win_rates(
    extracted_df: pd.DataFrame,
    grade_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    丸亀成績から当地勝率を算出する。

    着順点（当レース以前の成績から平均）:
      - 一般レース含む全レースに基本点を付与
      - SG競走 +2点、G1/G2競走 +1点
      - 各競走（一般/SG/G1/G2）の優勝戦はさらに +1点
    当地勝率 = 10 × √(平均着順点 ÷ 10)
    過去成績がなく算出できない場合は 0
    """
    work_df = attach_grade_info(extracted_df, grade_df)
    target_rates = work_df[LOCAL_RATE_META_COLS].copy()

    if history_df is not None and len(history_df) > 0:
        timeline = pd.concat([history_df[LOCAL_RATE_META_COLS], target_rates], ignore_index=True)
        target_len = len(target_rates)
    else:
        timeline = target_rates.copy()
        target_len = len(extracted_df)

    timeline = timeline.sort_values(
        ["開催日", "日目", "レース", "艇"]
    ).reset_index(drop=True)

    player_stats: dict[int, dict] = {}
    local_rates: list[float] = []

    for row in timeline.itertuples(index=False):
        toban = int(row.登番)
        rank = int(row.着)
        grade = normalize_grade(row.grade)
        is_championship = _as_bool(row.is_championship)
        stats = player_stats.setdefault(toban, _empty_player_stats())

        local_rates.append(_calc_win_rate(stats))

        points = calc_place_points(rank, grade, is_championship)
        if rank in PLACE_POINTS:
            _update_player_stats(stats, points)

    computed = extracted_df[RACE_ROW_KEY].copy()
    computed["当地勝率"] = local_rates[-target_len:]
    computed["当地勝率"] = computed["当地勝率"].fillna(0.0)

    rate_meta = work_df[LOCAL_RATE_META_COLS].copy()
    return computed, rate_meta


def compute(
    extracted_df: pd.DataFrame,
    player_records: list[dict],
    grade_df: pd.DataFrame,
    rate_history_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    抽出DataFrameと選手生データから計算列のみを生成する。

    Returns:
        (計算DataFrame[RACE_ROW_KEY + COMPUTED_COLUMNS], 当地勝率履歴用メタ)
    """
    local_df, rate_meta = compute_local_win_rates(
        extracted_df, grade_df, rate_history_df
    )
    player_computed_df = build_player_computed_df(player_records)
    player_df = attach_computed_player_fields(extracted_df, player_computed_df)

    computed = local_df.merge(
        player_df,
        on=RACE_ROW_KEY,
        how="left",
        validate="one_to_one",
    )
    return computed[RACE_ROW_KEY + COMPUTED_COLUMNS], rate_meta


DATASETS = [
    {"years": range(14, 25), "output": "丸亀学習用_選手データ.csv"},
    {"years": range(25, 26), "output": "丸亀テスト用_選手データ.csv"},
]


def main():
    """計算済み選手データをCSV出力（デバッグ・確認用）"""
    from ExtRD import extract_player_records

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n=== {dataset['output']} ===")
        records = extract_player_records(dataset["years"])
        df = build_player_computed_df(records)

        output_path = OUTPUT_DIR / dataset["output"]
        df.to_csv(output_path, index=False, encoding="UTF-8-sig")

        print("総件数:", len(df))
        if len(df) > 0:
            meta = pd.DataFrame(records)
            print("登番数:", df["登番"].nunique())
            print("年×期:", meta.groupby(["年", "期"]).size().to_dict())
            print("重複件数:", meta.duplicated(DEDUP_KEY).sum())
            print("保存:", output_path)


if __name__ == "__main__":
    main()
