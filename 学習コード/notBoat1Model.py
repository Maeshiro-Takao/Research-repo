import pandas as pd
import lightgbm as lgb
from pathlib import Path
from itertools import permutations

# ============================================================
# 設定
# ============================================================
BASE_DIR = Path("/Users/maeshirotakao/ボートデータ/丸亀データ")

TRAIN_PATH = BASE_DIR / "丸亀データ_学習用_1号艇以外1着.csv"
TEST_PATH  = BASE_DIR / "丸亀データ_テスト用_1号艇以外1着.csv"
RESULT_PATH = BASE_DIR / "結果_1号艇以外1着_top5.csv"

FEATURE_COLS = ["モーター", "展示", "スタート"]
TOP_N = 5


# ============================================================
# 読み込み
# ============================================================
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["開催日", "レース", "着", "艇", "登番"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} に必要な列がありません: {missing}")

    df["着"] = pd.to_numeric(df["着"], errors="coerce")
    df["艇"] = pd.to_numeric(df["艇"], errors="coerce")
    df["登番"] = pd.to_numeric(df["登番"], errors="coerce")
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["着"].isin([1, 2, 3, 4, 5, 6])].copy()
    df = df.dropna(subset=FEATURE_COLS).copy()

    df["着"] = df["着"].astype(int)
    df["艇"] = df["艇"].astype(int)
    df["レース"] = pd.to_numeric(df["レース"], errors="coerce").astype(int)

    df["race_key"] = df["開催日"].astype(str) + "_R" + df["レース"].astype(str)

    return df.reset_index(drop=True)


# ============================================================
# レース単位の横持ち化
# ============================================================
def build_race_level_dataset(df: pd.DataFrame):
    rows = []

    for race_key, grp in df.groupby("race_key", sort=False):
        grp = grp.sort_values("艇")

        row = {
            "race_key": race_key,
            "開催日": grp["開催日"].iloc[0],
            "レース": int(grp["レース"].iloc[0]),
        }

        for _, r in grp.iterrows():
            tei = int(r["艇"])
            row[f"艇{tei}_モーター"] = r["モーター"]
            row[f"艇{tei}_展示"] = r["展示"]
            #row[f"艇{tei}_進入"] = r["進入"]
            row[f"艇{tei}_スタート"] = r["スタート"]

        top3 = grp.sort_values("着").head(3)["艇"].astype(int).tolist()
        row["y1"] = top3[0] - 1
        row["y2"] = top3[1] - 1
        row["y3"] = top3[2] - 1

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 学習
# ============================================================
def train_multiclass_model(train_df, test_df, target_col, model_name):
    feature_cols = [c for c in train_df.columns if c.startswith("艇")]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    train_ds = lgb.Dataset(X_train, label=y_train)
    valid_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

    params = {
        "objective": "multiclass",
        "num_class": 6,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }

    print(f"\n学習中: {model_name}")
    model = lgb.train(
        params,
        train_ds,
        valid_sets=[valid_ds],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model, feature_cols


# ============================================================
# 候補生成
# ============================================================
def predict_top5_for_race(row, model1, model2, model3, feature_cols, top_n=5):
    X = pd.DataFrame([row[feature_cols].values], columns=feature_cols)

    p1 = model1.predict(X)[0]
    p2 = model2.predict(X)[0]
    p3 = model3.predict(X)[0]

    candidates = []
    for a, b, c in permutations([1, 2, 3, 4, 5, 6], 3):
        if a == 1:
            continue
        score = p1[a - 1] * p2[b - 1] * p3[c - 1]
        candidates.append((f"{a}-{b}-{c}", score))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return [x[0] for x in candidates[:top_n]]


# ============================================================
# 評価
# ============================================================
def evaluate(test_race_df, model1, model2, model3, feature_cols, top_n=5):
    results = []

    for _, row in test_race_df.iterrows():
        actual = f"{row['y1']+1}-{row['y2']+1}-{row['y3']+1}"
        preds = predict_top5_for_race(row, model1, model2, model3, feature_cols, top_n=top_n)
        hit = int(actual in preds)

        out = {
            "race_key": row["race_key"],
            "実際": actual,
            "的中": hit,
        }
        for i, p in enumerate(preds, start=1):
            out[f"予測{i}"] = p

        results.append(out)

    return pd.DataFrame(results)


# ============================================================
# メイン
# ============================================================
def main():
    print("CSVからデータ読み込み中...")
    train = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)

    train_race = build_race_level_dataset(train)
    test_race = build_race_level_dataset(test)

    print(f"train: {len(train_race):,} レース")
    print(f"test : {len(test_race):,} レース")

    model1, feature_cols = train_multiclass_model(train_race, test_race, "y1", "1着モデル")
    model2, _ = train_multiclass_model(train_race, test_race, "y2", "2着モデル")
    model3, _ = train_multiclass_model(train_race, test_race, "y3", "3着モデル")

    df_result = evaluate(test_race, model1, model2, model3, feature_cols, top_n=TOP_N)

    if len(df_result) == 0:
        print("評価対象レースがありません。")
        return

    hit_rate = df_result["的中"].mean()
    n_hits = int(df_result["的中"].sum())
    n_races = len(df_result)

    print(f"\n{'='*50}")
    print("1号艇1着以外側 Top5結果")
    print(f"予測レース数 : {n_races:,}")
    print(f"的中数       : {n_hits:,}")
    print(f"的中率       : {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"{'='*50}")

    df_result.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n結果保存完了: {RESULT_PATH}")
    print(df_result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()