import subprocess
from pathlib import Path

downloads = Path("/Users/maeshirotakao/Downloads")
base_dir = Path("/Users/maeshirotakao/ボートデータ/競走成績/14年")

def open_lzh(lzh_path: Path) -> None:
    subprocess.run(["open", str(lzh_path)], check=True)

for i in range(1, 32):
    fileNum = f"k1401{i:02d}"
    lzh_path = downloads / f"{fileNum}.lzh"

    # どこかの月にフォルダが存在するか？
    exists = False
    for j in range(1, 13):
        monthNum = f"{j:02d}月"
        target_dir = base_dir / monthNum / fileNum  # ← dirを上書きしない
        if target_dir.is_dir():
            print("already exists:", target_dir)
            exists = True
            break

    if exists:
        continue  # ← 見つかったら開かない

    if not lzh_path.exists():
        print("lzh not found:", lzh_path)
        continue

    open_lzh(lzh_path)
