from pathlib import Path
import shutil

src = Path("/Users/maeshirotakao/ボートデータ/競走成績/14年")

for i in range(1, 13):
    # 01月 形式と 1月 形式の両方を試す
    candidates = [src / f"{i:02d}月", src / f"{i}月"]
    target_dir = next((p for p in candidates if p.is_dir()), None)

    if target_dir is None:
        print("month dir not found:", i)
        continue

    # 1) 子フォルダ内の .TXT を月フォルダ直下へ移動
    for f in target_dir.rglob("*.TXT"):
        if f.is_file() and f.parent != target_dir:
            shutil.move(str(f), str(target_dir / f.name))

    # 2) 空フォルダだけ削除（下から順に）
    for d in sorted((p for p in target_dir.rglob("*") if p.is_dir()),
                    key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
