import pandas as pd
from pathlib import Path

# =========================
# 設定
# =========================
BASE_DIR = Path("/Users/maeshirotakao/ボートデータ/選手データ_Excel")

YEAR_START = 2014
YEAR_END = 2025

SHEET_NAME = "データ"

# 出力
OUT_XLSX = BASE_DIR / "選手データ_2014-2025_選手別シート.xlsx"
OUT_CSV = BASE_DIR / "選手データ_2014-2025_全員統合.csv"

# =========================
# 抽出項目
# =========================
TARGET_COLS = [
    "登番",
    "名前漢字",
    "名前カナ",
    "支部",
    "級",
    "身長",
    "体重",
    "勝率",
    "複勝率",
    "1着回数",
    "2着回数",
    "出走回数",
    "優出回数",
    "優勝回数",
    "平均スタートタイミング",
    "1コース進入回数",
    "1コース複勝率",
    "1コース平均スタートタイミング",
    "1コース平均スタート順位",
    "2コース進入回数",
    "2コース複勝率",
    "2コース平均スタートタイミング",
    "2コース平均スタート順位",
    "3コース進入回数",
    "3コース複勝率",
    "3コース平均スタートタイミング",
    "3コース平均スタート順位",
    "4コース進入回数",
    "4コース複勝率",
    "4コース平均スタートタイミング",
    "4コース平均スタート順位",
    "5コース進入回数",
    "5コース複勝率",
    "5コース平均スタートタイミング",
    "5コース平均スタート順位",
    "6コース進入回数",
    "6コース複勝率",
    "6コース平均スタートタイミング",
    "6コース平均スタート順位",
    "前期級",
    "前々期級",
    "前々々期級",
    "前期能力指数",
    "今期能力指数",
]

def read_one_xlsx(path: Path) -> pd.DataFrame | None:
    """1つのExcelファイルを読み込み（補正なし）"""
    if not path.exists():
        return None

    try:
        df = pd.read_excel(path, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"  [ERROR] {path.name}: {e}")
        return None

    if "登番" not in df.columns:
        return None

    # 取りたい列だけに絞る（無い列はNaNで補完）
    out = df.copy()
    for c in TARGET_COLS:
        if c not in out.columns:
            out[c] = None
    out = out[TARGET_COLS].copy()

    # 登番欠損行は除外
    out = out.dropna(subset=["登番"]).copy()

    return out

def year_to_yy(year: int) -> str:
    return str(year)[2:]

def main():
    print("="*70)
    print("選手データ一括抽出ツール（選手別シート版・補正なし）")
    print("="*70 + "\n")
    
    all_frames = []

    print("[1] ファイルを読み込み中...\n")
    for year in range(YEAR_START, YEAR_END + 1):
        yy = year_to_yy(year)
        for term in ["前期", "後期"]:
            path = BASE_DIR / f"{yy}年{term}.xlsx"
            print(f"  {year}年{term:2s}...", end="", flush=True)
            df = read_one_xlsx(path)
            if df is not None and len(df) > 0:
                print(f" ✓ ({len(df):,}行)")
                all_frames.append(df)
            else:
                print(f" ✗")

    if not all_frames:
        print("\n❌ データが抽出できませんでした")
        return

    print(f"\n[2] データを統合中...")
    full = pd.concat(all_frames, ignore_index=True)

    print(f"\n" + "="*70)
    print("✅ 抽出完了")
    print("="*70)
    print(f"  合計行数: {len(full):,}")
    print(f"  登番ユニーク数: {full['登番'].nunique():,}人")

    # CSV出力（全員統合版）
    print(f"\n[3] CSV（全員統合版）を出力中...")
    full.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✓ {OUT_CSV}")

    # Excel出力（選手別シート版）
    print(f"\n[4] Excel（選手別シート版）を生成中...")
    print(f"     ※選手数が多い場合、時間がかかります...")

    grouped = full.groupby('登番')
    total_players = len(grouped)
    processed = 0

    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as writer:
        for roulette_num, player_data in grouped:
            # シート名を作成（登番をシート名に使用）
            sheet_name = str(int(roulette_num))
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            
            player_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            processed += 1
            if processed % 100 == 0 or processed == total_players:
                print(f"     {processed}/{total_players}人処理済み...")

    print(f"  ✓ {OUT_XLSX}")
    print(f"     (全 {total_players}人分のシート)")

    print(f"\n" + "="*70)
    print("処理完了！")
    print("="*70)
    print(f"\n出力ファイル:")
    print(f"  ・CSV: {OUT_CSV}")
    print(f"  ・Excel: {OUT_XLSX}")
    print()

if __name__ == "__main__":
    main()