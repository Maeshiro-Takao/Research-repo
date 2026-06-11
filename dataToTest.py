import pandas as pd
from pathlib import Path

# ============================================================
# 設定
# ============================================================
INPUT_CSV = Path("丸亀データ/丸亀データ_テスト用_選手付き.csv")

OUT_1ST_BOAT1 = Path("丸亀データ/丸亀データ_テスト用_選手付き_1号艇1着.csv")
OUT_NOT_1ST_BOAT1 = Path("丸亀データ/丸亀データ_テスト用_選手付き_1号艇以外1着.csv")

# ============================================================
# 読み込み
# ============================================================
df = pd.read_csv(INPUT_CSV)

# 必須列チェック
required_cols = ["開催日", "レース", "艇", "着"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"必要な列が不足: {missing}")

# 数値化
df["着"] = pd.to_numeric(df["着"], errors="coerce")
df["艇"] = pd.to_numeric(df["艇"], errors="coerce")

# ============================================================
# レース単位で「1号艇が1着か」を判定
# ============================================================
def is_boat1_win(race_df):
    """
    そのレースで1号艇が1着なら1、それ以外は0
    """
    cond = (race_df["艇"] == 1) & (race_df["着"] == 1)
    return int(cond.any())

race_flag = (
    df.groupby(["開催日", "レース"])
      .apply(is_boat1_win)
      .reset_index(name="boat1_win")
)

# ============================================================
# 元データにフラグを結合
# ============================================================
df = df.merge(race_flag, on=["開催日", "レース"], how="left")

# ============================================================
# 分割
# ============================================================
df_boat1_win = df[df["boat1_win"] == 1].copy()
df_not_boat1_win = df[df["boat1_win"] == 0].copy()

# ============================================================
# 保存
# ============================================================
df_boat1_win.to_csv(OUT_1ST_BOAT1, index=False, encoding="utf-8-sig")
df_not_boat1_win.to_csv(OUT_NOT_1ST_BOAT1, index=False, encoding="utf-8-sig")

# ============================================================
# 確認
# ============================================================
print("==== 分割結果 ====")
print(f"全体: {len(df):,} 行")

print("\n【1号艇1着】")
print(f"行数: {len(df_boat1_win):,}")
print(f"レース数: {df_boat1_win[['開催日','レース']].drop_duplicates().shape[0]:,}")

print("\n【1号艇以外1着】")
print(f"行数: {len(df_not_boat1_win):,}")
print(f"レース数: {df_not_boat1_win[['開催日','レース']].drop_duplicates().shape[0]:,}")

print("\n保存完了")
print(OUT_1ST_BOAT1)
print(OUT_NOT_1ST_BOAT1)