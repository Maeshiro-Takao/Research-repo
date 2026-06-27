"""
ExtRD.py で抽出した DataFrame から学習用特徴量を計算する。

勝率 = 着順点合計 / 出走数（SG+2, G1/G2+1, 優勝戦+1）
2連率/3連率 = 連対率（%）

算出期間:
  全国: 開催初日の月を含む過去6ヶ月（今節除外）〜前検日前日
  当地: 開催初日月の24ヶ月前の年1/1（今節除外）〜前検日前日（12-13年成績含む）
  モーター: 11月始まりの年度内・使用開始から最大1年（今節除外）
  ボート: 7月始まりの年度内・使用開始から最大1年（今節除外）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ExtRD import (
    MARUGAME_CODE,
    RACE_ROW_KEY,
    extract_history_dataframe,
)

RACE_KEY = ["開催日", "日目", "レース"]

NATIONAL_COLS = ["勝率", "2連率", "3連率"]
LOCAL_COLS = ["当地勝率", "当地2連率", "当地3連率"]
MOTOR_COLS = ["モーター勝率", "モーター2連率", "モーター3連率"]
BOAT_COLS = ["ボート勝率", "ボート2連率", "ボート3連率"]
PERIOD_COLS = ["算出期間自", "算出期間至"]

COMPUTED_COLUMNS = NATIONAL_COLS + LOCAL_COLS + MOTOR_COLS + BOAT_COLS + PERIOD_COLS

PLACE_POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
LOCAL_RATE_META_COLS = [
    "開催日", "日目", "レース", "艇", "登番", "着", "grade", "is_championship",
]

MOTOR_RESET_MONTH = 11
BOAT_RESET_MONTH = 7
LOCAL_HISTORY_START_YEAR = 12

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "編集データ"


def round_rate(value: float | None, decimals: int) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return round(float(value), decimals)


def normalize_grade(grade: str | None) -> str:
    if grade is None or pd.isna(grade):
        return "一般"
    text = str(grade).strip().upper()
    if text in {"SG", "G1", "G2"}:
        return text
    return "一般"


def grade_point_bonus(grade: str | None) -> int:
    g = normalize_grade(grade)
    if g == "SG":
        return 2
    if g in {"G1", "G2"}:
        return 1
    return 0


def championship_point_bonus(is_championship: bool) -> int:
    return 1 if is_championship else 0


def calc_place_points(rank: int, grade: str | None, is_championship: bool) -> int:
    if rank not in PLACE_POINTS:
        return 0
    return (
        PLACE_POINTS[rank]
        + grade_point_bonus(grade)
        + championship_point_bonus(is_championship)
    )


def attach_grade_info(
    race_df: pd.DataFrame,
    grade_df: pd.DataFrame,
) -> pd.DataFrame:
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


def meet_opening_date(race_date: pd.Timestamp, day_number: int) -> pd.Timestamp:
    return race_date - pd.Timedelta(days=int(day_number) - 1)


def zenken_prev_day(opening_date: pd.Timestamp) -> pd.Timestamp:
    """前検日前日 = 開催初日の2日前"""
    return opening_date - pd.Timedelta(days=2)


def national_period(opening_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    opening_month = opening_date.to_period("M")
    start = (opening_month - 5).to_timestamp()
    end = zenken_prev_day(opening_date)
    return start, end


def local_period(opening_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    year = (opening_date - pd.DateOffset(months=24)).year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = zenken_prev_day(opening_date)
    return start, end


def equipment_season_start(race_date: pd.Timestamp, reset_month: int) -> pd.Timestamp:
    """モーター=11月、ボート=7月を境に年度を区切る"""
    if race_date.month >= reset_month:
        return pd.Timestamp(race_date.year, reset_month, 1)
    return pd.Timestamp(race_date.year - 1, reset_month, 1)


def equipment_period(
    usage_start: pd.Timestamp,
    season_start: pd.Timestamp,
    opening_date: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(usage_start, season_start)
    end = min(
        zenken_prev_day(opening_date),
        usage_start + pd.Timedelta(days=365) - pd.Timedelta(days=1),
    )
    return start, end


def compute_win_and_ren_rates(
    ranks: np.ndarray,
    points: np.ndarray | None = None,
) -> tuple[float, float, float]:
    runs = len(ranks)
    if runs <= 0:
        return 0.0, 0.0, 0.0

    top2 = int(np.sum(ranks <= 2))
    top3 = int(np.sum(ranks <= 3))

    if points is not None and len(points) > 0:
        win_rate = float(np.sum(points)) / runs
    else:
        win_rate = int(np.sum(ranks == 1)) / runs * 10

    return (
        round_rate(win_rate, 2),
        round_rate(top2 / runs * 100, 1),
        round_rate(top3 / runs * 100, 1),
    )


def _merge_grade_columns(df: pd.DataFrame, grade_df: pd.DataFrame) -> pd.DataFrame:
    grade_cols = grade_df.rename(
        columns={"グレード": "grade", "優勝戦": "is_championship"}
    ).copy()
    grade_cols["開催日"] = pd.to_datetime(grade_cols["開催日"])
    merged = df.merge(
        grade_cols[["開催日", "日目", "レース", "grade", "is_championship"]],
        on=RACE_KEY,
        how="left",
    )
    merged["grade"] = merged["grade"].map(normalize_grade)
    merged["is_championship"] = merged["is_championship"].fillna(False).astype(bool)
    return merged


def _prepare_history(history_df: pd.DataFrame) -> pd.DataFrame:
    df = history_df.copy()
    df["開催日"] = pd.to_datetime(df["開催日"])
    df["日目"] = df["日目"].astype(int)
    df["登番"] = df["登番"].astype(int)
    df["着"] = df["着"].astype(int)
    df["opening_date"] = df.apply(
        lambda r: meet_opening_date(r["開催日"], r["日目"]),
        axis=1,
    )
    return df.sort_values(["開催日", "登番"]).reset_index(drop=True)


def _add_points_column(df: pd.DataFrame, *, with_grade: bool) -> pd.DataFrame:
    out = df.copy()
    if with_grade:
        out["points"] = out.apply(
            lambda r: calc_place_points(int(r["着"]), r["grade"], bool(r["is_championship"])),
            axis=1,
        )
    else:
        out["points"] = out["着"].map(PLACE_POINTS).fillna(0).astype(int)
    return out


def _prepare_national_history(
    history_df: pd.DataFrame,
    grade_df: pd.DataFrame,
) -> pd.DataFrame:
    df = _prepare_history(history_df)
    df = _merge_grade_columns(df, grade_df)
    df["points"] = df.apply(
        lambda r: calc_place_points(
            int(r["着"]),
            r["grade"] if r["場"] == MARUGAME_CODE else "一般",
            bool(r["is_championship"]) if r["場"] == MARUGAME_CODE else False,
        ),
        axis=1,
    )
    return df


def _build_local_history(national_hist: pd.DataFrame, grade_df: pd.DataFrame) -> pd.DataFrame:
    """丸亀成績（12-13年含む）から当地勝率用履歴を構築"""
    del grade_df
    mg = national_hist.loc[national_hist["場"] == MARUGAME_CODE].copy()
    return mg.sort_values(["開催日", "登番"]).reset_index(drop=True)


def _first_usage_in_season(
    history: pd.DataFrame,
    key_col: str,
    reset_month: int,
) -> dict[tuple, pd.Timestamp]:
    usage: dict[tuple, pd.Timestamp] = {}
    for row in history.sort_values("開催日").itertuples(index=False):
        season_start = equipment_season_start(row.開催日, reset_month)
        key = (row.場, int(getattr(row, key_col)), season_start)
        if key not in usage:
            usage[key] = row.開催日
    return usage


def _filter_window_arrays(
    dates: np.ndarray,
    openings: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    exclude_opening: pd.Timestamp | None = None,
    is_marugame: np.ndarray | None = None,
) -> np.ndarray:
    mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    if exclude_opening is not None and is_marugame is not None:
        mask &= ~(
            (openings == np.datetime64(exclude_opening)) & is_marugame
        )
    return mask


def _group_arrays(group: pd.DataFrame | None, *, local: bool = False):
    if group is None or len(group) == 0:
        return None
    is_mg = np.ones(len(group), dtype=bool) if local else (group["場"] == MARUGAME_CODE).to_numpy()
    return {
        "dates": group["開催日"].to_numpy(dtype="datetime64[ns]"),
        "openings": group["opening_date"].to_numpy(dtype="datetime64[ns]"),
        "ranks": group["着"].to_numpy(dtype=int),
        "points": group["points"].to_numpy(dtype=float),
        "is_marugame": is_mg,
    }


def compute_rates_for_targets(
    targets: pd.DataFrame,
    national_hist: pd.DataFrame,
    local_hist: pd.DataFrame,
    motor_usage: dict[tuple, pd.Timestamp],
    boat_usage: dict[tuple, pd.Timestamp],
) -> pd.DataFrame:
    nat_groups = {
        int(t): _group_arrays(g.sort_values("開催日").reset_index(drop=True))
        for t, g in national_hist.groupby("登番")
    }
    loc_groups = {
        int(t): _group_arrays(g.sort_values("開催日").reset_index(drop=True), local=True)
        for t, g in local_hist.groupby("登番")
    }
    motor_groups = {
        int(m): _group_arrays(g.sort_values("開催日").reset_index(drop=True), local=True)
        for m, g in local_hist.groupby("モーター")
    }
    boat_groups = {
        int(b): _group_arrays(g.sort_values("開催日").reset_index(drop=True), local=True)
        for b, g in local_hist.groupby("ボート")
    }

    rows: list[dict] = []
    for row in targets.itertuples(index=False):
        race_date = pd.Timestamp(row.開催日)
        day_number = int(row.日目)
        toban = int(row.登番)
        motor = int(row.モーター)
        boat = int(row.ボート)
        opening = meet_opening_date(race_date, day_number)

        nat_start, nat_end = national_period(opening)
        loc_start, loc_end = local_period(opening)

        nat = nat_groups.get(toban)
        nat_mask = (
            _filter_window_arrays(
                nat["dates"], nat["openings"], nat_start, nat_end,
                exclude_opening=opening, is_marugame=nat["is_marugame"],
            )
            if nat is not None
            else np.array([], dtype=bool)
        )

        loc = loc_groups.get(toban)
        loc_mask = (
            _filter_window_arrays(
                loc["dates"], loc["openings"], loc_start, loc_end,
                exclude_opening=opening, is_marugame=loc["is_marugame"],
            )
            if loc is not None
            else np.array([], dtype=bool)
        )

        motor_season = equipment_season_start(race_date, MOTOR_RESET_MONTH)
        motor_key = (MARUGAME_CODE, motor, motor_season)
        motor_start = motor_usage.get(motor_key, race_date)
        mot_start, mot_end = equipment_period(motor_start, motor_season, opening)
        mot = motor_groups.get(motor)
        mot_mask = (
            _filter_window_arrays(
                mot["dates"], mot["openings"], mot_start, mot_end,
                exclude_opening=opening, is_marugame=mot["is_marugame"],
            )
            if mot is not None
            else np.array([], dtype=bool)
        )

        boat_season = equipment_season_start(race_date, BOAT_RESET_MONTH)
        boat_key = (MARUGAME_CODE, boat, boat_season)
        boat_start = boat_usage.get(boat_key, race_date)
        boat_start_d, boat_end_d = equipment_period(boat_start, boat_season, opening)
        bt = boat_groups.get(boat)
        boat_mask = (
            _filter_window_arrays(
                bt["dates"], bt["openings"], boat_start_d, boat_end_d,
                exclude_opening=opening, is_marugame=bt["is_marugame"],
            )
            if bt is not None
            else np.array([], dtype=bool)
        )

        win, r2, r3 = compute_win_and_ren_rates(
            nat["ranks"][nat_mask] if nat_mask.any() else np.array([]),
            nat["points"][nat_mask] if nat_mask.any() else None,
        )
        loc_win, loc_r2, loc_r3 = compute_win_and_ren_rates(
            loc["ranks"][loc_mask] if loc_mask.any() else np.array([]),
            loc["points"][loc_mask] if loc_mask.any() else None,
        )
        mot_win, mot_r2, mot_r3 = compute_win_and_ren_rates(
            mot["ranks"][mot_mask] if mot_mask.any() else np.array([]),
            mot["points"][mot_mask] if mot_mask.any() else None,
        )
        boat_win, boat_r2, boat_r3 = compute_win_and_ren_rates(
            bt["ranks"][boat_mask] if boat_mask.any() else np.array([]),
            bt["points"][boat_mask] if boat_mask.any() else None,
        )

        rows.append(
            {
                "開催日": row.開催日.strftime("%Y-%m-%d")
                if hasattr(row.開催日, "strftime")
                else row.開催日,
                "日目": row.日目,
                "レース": row.レース,
                "艇": row.艇,
                "勝率": win,
                "2連率": r2,
                "3連率": r3,
                "当地勝率": loc_win,
                "当地2連率": loc_r2,
                "当地3連率": loc_r3,
                "モーター勝率": mot_win,
                "モーター2連率": mot_r2,
                "モーター3連率": mot_r3,
                "ボート勝率": boat_win,
                "ボート2連率": boat_r2,
                "ボート3連率": boat_r3,
                "算出期間自": nat_start.strftime("%Y-%m-%d"),
                "算出期間至": nat_end.strftime("%Y-%m-%d"),
            }
        )

    return pd.DataFrame(rows)


def compute_local_win_rates(
    extracted_df: pd.DataFrame,
    grade_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del history_df
    work_df = attach_grade_info(extracted_df, grade_df)
    rate_meta = work_df[LOCAL_RATE_META_COLS].copy()
    return pd.DataFrame(columns=RACE_ROW_KEY + ["当地勝率"]), rate_meta


def compute(
    extracted_df: pd.DataFrame,
    player_records: list[dict],
    grade_df: pd.DataFrame,
    rate_history_df: pd.DataFrame | None = None,
    history_years: range | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del player_records, rate_history_df

    work_df = attach_grade_info(extracted_df, grade_df)
    rate_meta = work_df[LOCAL_RATE_META_COLS].copy()

    if history_years is None:
        years = pd.to_datetime(extracted_df["開催日"]).dt.year
        min_y = max(LOCAL_HISTORY_START_YEAR, int(years.min()) - 2)
        max_y = int(years.max())
        history_years = range(min_y % 100, max_y % 100 + 1)
    else:
        start = min(history_years.start, LOCAL_HISTORY_START_YEAR)
        history_years = range(start, history_years.stop)

    print(f"    全国履歴読込: {history_years.start + 2000}〜{history_years.stop - 1 + 2000}年...")
    national_raw = extract_history_dataframe(history_years)
    if len(national_raw) == 0:
        raise ValueError("全国成績履歴が空です。競走成績TXTを確認してください。")

    national_hist = _prepare_national_history(national_raw, grade_df)
    local_hist = _build_local_history(national_hist, grade_df)

    motor_usage = _first_usage_in_season(local_hist, "モーター", MOTOR_RESET_MONTH)
    boat_usage = _first_usage_in_season(local_hist, "ボート", BOAT_RESET_MONTH)

    targets = work_df[
        RACE_ROW_KEY + ["登番", "モーター", "ボート"]
    ].copy()
    targets["開催日"] = pd.to_datetime(targets["開催日"])

    computed = compute_rates_for_targets(
        targets,
        national_hist,
        local_hist,
        motor_usage,
        boat_usage,
    )

    return computed[RACE_ROW_KEY + COMPUTED_COLUMNS], rate_meta


def main():
    print("CalcRD: 成績履歴ベースの勝率計算は ComRD.py 経由で実行してください。")


if __name__ == "__main__":
    main()
