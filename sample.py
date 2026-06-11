import re
from pathlib import Path

base_dir = Path("/Users/maeshirotakao/ボートデータ/競走成績/25年")

for txt_file in base_dir.rglob("*.TXT"):
    print("processing:", txt_file)

    try:
        text = txt_file.read_text(encoding="cp932", errors="ignore")
    except Exception:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")

    new_lines = []

    for line in text.splitlines():
        # 全角スペース→半角スペース
        line = line.replace("　", " ")

        # 連続スペース・タブを1個に圧縮
        line = re.sub(r"[ \t]+", " ", line)

        # 行頭・行末の空白削除
        line = line.strip()

        new_lines.append(line)

    txt_file.write_text(
        "\n".join(new_lines),
        encoding="utf-8"
    )

print("完了")