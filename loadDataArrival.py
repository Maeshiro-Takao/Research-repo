from pathlib import Path
import re
import openpyxl
import pandas as pd
from datetime import datetime

commonPath = Path("/Users/maeshirotakao/ボートデータ/競走成績_Excel")
outputPath = Path("/Users/maeshirotakao/ボートデータ/競走成績_csv")
out_dir = outputPath
out_dir.mkdir(exist_ok=True)

raceTrack = ["びわこ","芦屋","下関","蒲郡","丸亀","宮島","桐生","戸田","江戸川","三国",
             "児島","若松","住之江","常滑","多摩川","大村","津","唐津","徳山","尼崎",
             "浜名湖","福岡","平和島","鳴門"]

race_pat = re.compile(r'【\s*(\d+)\s*R】')

# 日付フィルタリング用
START_DATE = datetime(2014, 1, 1)
END_DATE = datetime(2024, 12, 31)

def to_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return int(s)
    return None

def parse_date(date_str):
    """日付文字列をdatetimeに変換"""
    if date_str is None:
        return None
    
    if isinstance(date_str, datetime):
        return date_str
    
    if isinstance(date_str, str):
        date_str = date_str.strip()
        # YYYY/MM/DD形式を試す
        try:
            return datetime.strptime(date_str, "%Y/%m/%d")
        except:
            pass
        # YYYY-MM-DD形式を試す
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            pass
    
    return None

def is_date_in_range(date_str):
    """日付が指定範囲内かチェック"""
    date = parse_date(date_str)
    if date is None:
        return False
    return START_DATE <= date <= END_DATE

def extract_top3(xlsx_path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)

    rows = []
    for sheetname in wb.sheetnames:
        # シート名が日付形式かチェック
        if not is_date_in_range(sheetname):
            continue
        
        ws = wb[sheetname]
        current_race = None
        in_results = False

        # 1列目=「着」、2列目=「艇」を走査（びわこ.xlsxと同じ前提）
        for a, b in ws.iter_rows(min_col=1, max_col=2, values_only=True):
            # レース番号行
            if isinstance(a, str):
                s = a.strip()
                m = race_pat.match(s)
                if m:
                    current_race = int(m.group(1))
                    in_results = False
                    continue
                # ヘッダー行（着 / 艇）
                if s == "着" and isinstance(b, str) and b.strip() == "艇":
                    in_results = True
                    continue

            if not in_results:
                continue

            place = to_int(a)
            lane = to_int(b)

            # 1〜6着を取得
            if place in (1, 2, 3, 4, 5, 6) and lane is not None:
                rows.append({"日付": sheetname, "R": current_race, "着": place, "艇": lane})

    return pd.DataFrame(rows, columns=["日付", "R", "着", "艇"])

def main():
    print("="*70)
    print("レース成績データ抽出ツール")
    print("="*70 + "\n")
    
    total_rows = 0
    
    for name in raceTrack:
        xlsx_path = commonPath / f"{name}.xlsx"
        if not xlsx_path.exists():
            print(f"❌ MISSING: {xlsx_path}")
            continue

        try:
            df = extract_top3(xlsx_path)
            
            if len(df) == 0:
                print(f"⚠️  {name:8s}: データなし")
                continue
            
            out_path = out_dir / f"{name}arrivalData.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            
            total_rows += len(df)
            print(f"✓ {name:8s}: {len(df):,}行 → {out_path.name}")
        except Exception as e:
            print(f"❌ FAILED: {xlsx_path} -> {repr(e)}")

    print()
    print("="*70)
    print(f"合計: {total_rows:,}行")
    print(f"出力先: {out_dir}")
    print("="*70)

if __name__ == "__main__":
    main()