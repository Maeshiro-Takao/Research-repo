import pandas as pd
from pathlib import Path
import glob

# ============================================================
# 設定 ← ここだけ環境に合わせて変更
# ============================================================
# fanYYMM.txtが置いてあるフォルダ（サブフォルダも自動検索）
INPUT_DIR = Path("/Users/maeshirotakao/ボートデータ/選手データ")

# 出力CSVのパス
OUTPUT_CSV = INPUT_DIR / "選手データ_2026.csv"

# ============================================================
# フィールド定義（バイト単位）
# ============================================================
FIELDS = [
    ('登番',       4),
    ('名前漢字',  16),
    ('名前カナ',  15),
    ('支部',       4),
    ('級',         2),
    ('年号',       1),
    ('生年月日',   6),
    ('性別',       1),
    ('年齢',       2),
    ('身長',       3),
    ('体重',       2),
    ('血液型',     2),
    ('勝率',       4),
    ('複勝率',     4),
    ('1着回数',    3),
    ('2着回数',    3),
    ('出走回数',   3),
    ('優出回数',   2),
    ('優勝回数',   2),
    ('平均ST',     3),
    ('1コース進入回数', 3), ('1コース複勝率', 4), ('1コースST', 3), ('1コースST順位', 3),
    ('2コース進入回数', 3), ('2コース複勝率', 4), ('2コースST', 3), ('2コースST順位', 3),
    ('3コース進入回数', 3), ('3コース複勝率', 4), ('3コースST', 3), ('3コースST順位', 3),
    ('4コース進入回数', 3), ('4コース複勝率', 4), ('4コースST', 3), ('4コースST順位', 3),
    ('5コース進入回数', 3), ('5コース複勝率', 4), ('5コースST', 3), ('5コースST順位', 3),
    ('6コース進入回数', 3), ('6コース複勝率', 4), ('6コースST', 3), ('6コースST順位', 3),
    ('前期級',       2),
    ('前々期級',     2),
    ('前々々期級',   2),
    ('前期能力指数', 4),
    ('今期能力指数', 4),
    ('年',           4),
    ('期',           1),
    ('算出期間（自）', 8),
    ('算出期間（至）', 8),
    ('養成期',       3),
]

MIN_RECORD_BYTES = sum(s for _, s in FIELDS)  # 198bytes

# ============================================================
# 出力する列（この列だけCSVに保存）
# ============================================================
OUTPUT_COLS = [
    '年月', '登番',
    '勝率', '複勝率',
    '1着回数', '2着回数', '出走回数', '優出回数', '優勝回数',
    '平均ST',
    '1コース進入回数', '1コース複勝率', '1コースST', '1コースST順位',
    '2コース進入回数', '2コース複勝率', '2コースST', '2コースST順位',
    '3コース進入回数', '3コース複勝率', '3コースST', '3コースST順位',
    '4コース進入回数', '4コース複勝率', '4コースST', '4コースST順位',
    '5コース進入回数', '5コース複勝率', '5コースST', '5コースST順位',
    '6コース進入回数', '6コース複勝率', '6コースST', '6コースST順位',
]

# ÷100して小数にするフィールド
RATE_FIELDS = {
    '勝率', '複勝率',
    '1コース複勝率', '2コース複勝率', '3コース複勝率',
    '4コース複勝率', '5コース複勝率', '6コース複勝率',
    '平均ST',
    '1コースST', '2コースST', '3コースST',
    '4コースST', '5コースST', '6コースST',
}

# 整数にするフィールド
INT_FIELDS = {
    '登番', '年齢', '身長', '体重',
    '1着回数', '2着回数', '出走回数', '優出回数', '優勝回数',
    '1コース進入回数', '2コース進入回数', '3コース進入回数',
    '4コース進入回数', '5コース進入回数', '6コース進入回数',
    '1コースST順位', '2コースST順位', '3コースST順位',
    '4コースST順位', '5コースST順位', '6コースST順位',
    '前期能力指数', '今期能力指数', '年', '養成期',
}


# ============================================================
# パース処理
# ============================================================
def parse_file(filepath: Path) -> list[dict]:
    """1つのfanYYMM.txtを読み込んでレコードのリストを返す"""
    with open(filepath, 'rb') as f:
        data = f.read()

    # ファイル名から年月を取得（例: fan1401.txt → 年月='2014-01'）
    stem = filepath.stem  # 'fan1401'
    yymm = stem.replace('fan', '')  # '1401'
    if len(yymm) == 4:
        yy = int(yymm[:2])
        mm = int(yymm[2:])
        year = 2000 + yy
        label = f"{year}-{mm:02d}"
    else:
        label = stem

    records = []
    for line in data.split(b'\r\n'):
        if len(line) < MIN_RECORD_BYTES:
            continue

        record = {'年月': label}
        pos = 0

        for name, size in FIELDS:
            raw = line[pos:pos + size].decode('cp932', errors='replace').strip()

            if name in RATE_FIELDS:
                try:
                    record[name] = round(int(raw) / 100, 2)
                except ValueError:
                    record[name] = None
            elif name in INT_FIELDS:
                try:
                    record[name] = int(raw)
                except ValueError:
                    record[name] = None
            else:
                record[name] = raw

            pos += size

        records.append(record)

    return records


# ============================================================
# メイン
# ============================================================
def main():
    # fanYYMM.txt を再帰的に検索してソート（14年/fan1401.txt 形式に対応）
    txt_files = sorted(INPUT_DIR.rglob('fan[0-9][0-9][0-9][0-9].txt'))
    
    # 見つからない場合は「14年」「15年」フォルダを直接確認
    if not txt_files:
        txt_files = sorted(INPUT_DIR.glob('[0-9][0-9]年/fan[0-9][0-9][0-9][0-9].txt'))

    if not txt_files:
        print(f"エラー: {INPUT_DIR} 配下にfanYYMM.txtが見つかりません")
        return

    print(f"対象ファイル数: {len(txt_files)}件")

    all_records = []
    for i, filepath in enumerate(txt_files, 1):
        try:
            records = parse_file(filepath)
            all_records.extend(records)
            print(f"[{i:3d}/{len(txt_files)}] {filepath.name}: {len(records)}件")
        except Exception as e:
            print(f"[{i:3d}/{len(txt_files)}] {filepath.name}: エラー → {e}")

    if not all_records:
        print("変換できるレコードがありませんでした")
        return

    df = pd.DataFrame(all_records)

    # 出力列のみに絞り込む
    df = df[[c for c in OUTPUT_COLS if c in df.columns]]

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n{'='*50}")
    print(f"完了: {len(df):,}件のレコード")
    print(f"期間: {df['年月'].min()} 〜 {df['年月'].max()}")
    print(f"出力: {OUTPUT_CSV}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()