"""
期待値判定の実行戦略（Execution Strategy）と市場オッズ取得。

- 市場オッズ: 現状は5分前オッズ列を参照し、欠損時は締切時オッズへフォールバック
- 参加判定: ルールベース（機械学習モデルは使用しない）
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RACE_KEY = ["開催日", "日目", "レース"]

# 取得したいオッズ時点（将来5分前データが揃ったら DEFAULT のまま切り替え可能）
DEFAULT_ODDS_TIMING = "5分前"

ODDS_COLUMN_MAP: dict[str, str] = {
    "10分前": "10分前オッズ",
    "5分前": "5分前オッズ",
    "2分前": "2分前オッズ",
    "締切時": "締切時オッズ",
}

# 5分前オッズが未整備の間は締切時オッズを代用
FALLBACK_ODDS_TIMING = "締切時"

MIN_MARKET_COMBOS = 120


@dataclass
class ExecutionConfig:
    """実行戦略のパラメータ（ルール変更時はここを編集）"""

    top_n: int = 15
    bet_yen: int = 100
    roi_threshold: float = 1.0  # 1.0 = 100%


def combo_str(combo: tuple[int, int, int]) -> str:
    return f"{combo[0]}-{combo[1]}-{combo[2]}"


def resolve_odds_columns(
    timing: str = DEFAULT_ODDS_TIMING,
    fallback_timing: str = FALLBACK_ODDS_TIMING,
) -> tuple[str, str]:
    """主オッズ列とフォールバック列を返す"""
    primary = ODDS_COLUMN_MAP.get(timing, ODDS_COLUMN_MAP[DEFAULT_ODDS_TIMING])
    fallback = ODDS_COLUMN_MAP.get(fallback_timing, ODDS_COLUMN_MAP[FALLBACK_ODDS_TIMING])
    return primary, fallback


def pick_market_odds(row: pd.Series, primary_col: str, fallback_col: str) -> float:
    """5分前オッズを優先し、欠損なら締切時オッズを返す"""
    for col in (primary_col, fallback_col):
        if col not in row.index:
            continue
        val = row[col]
        if pd.notna(val) and float(val) > 0:
            return float(val)
    return float("nan")


def _read_orange_buoy_csv(path: Path) -> pd.DataFrame:
    """末尾カンマ等の不正行を吸収してオッズ保管庫CSVを読み込む"""
    rows: list[list[str]] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)
        for fields in reader:
            if not fields:
                continue
            if len(fields) > ncols:
                fields = fields[:ncols]
            elif len(fields) < ncols:
                fields = fields + [""] * (ncols - len(fields))
            rows.append(fields)
    return pd.DataFrame(rows, columns=header)


def load_market_odds(
    path: Path,
    race_schedule_df: pd.DataFrame | None = None,
    timing: str = DEFAULT_ODDS_TIMING,
    fallback_timing: str = FALLBACK_ODDS_TIMING,
) -> pd.DataFrame:
    """
    市場オッズ（120通り/レース）を読み込む。

    戻り値: RACE_KEY + 3連単 + 市場オッズ
    """
    if not path.exists():
        raise FileNotFoundError(f"市場オッズCSVが見つかりません: {path}")

    raw = _read_orange_buoy_csv(path)
    primary_col, fallback_col = resolve_odds_columns(timing, fallback_timing)

    for col in ODDS_COLUMN_MAP.values():
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.rename(columns={"組み合わせ": "3連単"})
    raw["市場オッズ"] = raw.apply(
        lambda row: pick_market_odds(row, primary_col, fallback_col),
        axis=1,
    )
    raw["開催日"] = raw["開催日"].astype(str)
    raw["レース"] = pd.to_numeric(raw["レース"], errors="coerce").astype("Int64")

    work = raw.loc[raw["市場オッズ"].notna() & (raw["市場オッズ"] > 0)].copy()
    work = work[["開催日", "レース", "3連単", "市場オッズ"]]

    if race_schedule_df is not None:
        schedule = race_schedule_df[RACE_KEY].drop_duplicates().copy()
        schedule["開催日"] = schedule["開催日"].astype(str)
        schedule["レース"] = schedule["レース"].astype(int)
        work["レース"] = work["レース"].astype(int)
        work = work.merge(schedule, on=["開催日", "レース"], how="inner")

    return work[[*RACE_KEY, "3連単", "市場オッズ"]].reset_index(drop=True)


def load_payout_odds(path: Path) -> pd.DataFrame:
    """払戻オッズ（的中3連単1行/レース）を読み込む"""
    if not path.exists():
        raise FileNotFoundError(f"払戻オッズCSVが見つかりません: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = [*RACE_KEY, "3連単", "3連単オッズ"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"払戻オッズCSVに必要列がありません: {missing}")

    df["開催日"] = df["開催日"].astype(str)
    df["日目"] = df["日目"].astype(int)
    df["レース"] = df["レース"].astype(int)
    df["3連単オッズ"] = pd.to_numeric(df["3連単オッズ"], errors="coerce")
    return df[required].dropna(subset=["3連単オッズ"]).reset_index(drop=True)


def build_payout_map(payout_df: pd.DataFrame) -> dict[tuple, tuple[str, float]]:
    """レースキー → (的中3連単, 払戻オッズ)"""
    mapping: dict[tuple, tuple[str, float]] = {}
    for _, row in payout_df.iterrows():
        key = (row["開催日"], int(row["日目"]), int(row["レース"]))
        mapping[key] = (str(row["3連単"]), float(row["3連単オッズ"]))
    return mapping


def compute_expected_values(
    prob_map: dict[tuple[int, int, int], float],
    market_odds_map: dict[str, float],
) -> pd.DataFrame:
    """全120通り（利用可能な組み合わせ）について期待値 = 予測確率 × 市場オッズ"""
    rows: list[dict] = []
    for combo, prob in prob_map.items():
        label = combo_str(combo)
        odds = market_odds_map.get(label)
        if odds is None or pd.isna(odds) or odds <= 0:
            continue
        rows.append({
            "3連単": label,
            "予測確率": prob,
            "市場オッズ": float(odds),
            "期待値": prob * float(odds),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("期待値", ascending=False).reset_index(drop=True)


def should_bet(
    candidates: pd.DataFrame,
    config: ExecutionConfig | None = None,
) -> tuple[bool, pd.DataFrame, float]:
    """
    ルールベース参加判定。

    - 期待値上位 top_n 点を各 bet_yen 円購入
    - 上位点の平均期待値（= 平均期待ROI）が roi_threshold 以上なら参加
    """
    cfg = config or ExecutionConfig()
    if candidates.empty:
        return False, pd.DataFrame(), 0.0

    top = candidates.head(cfg.top_n).copy()
    avg_expected_roi = float(top["期待値"].mean())
    participate = avg_expected_roi >= cfg.roi_threshold
    return participate, top, avg_expected_roi
