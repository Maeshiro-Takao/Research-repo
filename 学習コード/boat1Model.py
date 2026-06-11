import pandas as pd
import lightgbm as lgb
from pathlib import Path

# ============================================================
# 設定
# ============================================================
BASE_DIR = Path("/Users/maeshirotakao/ボートデータ/丸亀データ")

TRAIN_PATH = BASE_DIR / "丸亀データ_学習用_1号艇1着.csv"
TEST_PATH  = BASE_DIR / "丸亀データ_テスト用_1号艇1着.csv"
RESULT_PATH = BASE_DIR / "結果_1号艇1着_top5.csv"

FEATURES = ["艇", "モーター", "展示", "スタート"]
TOP_N = 5


# ============================================================
# CSV読み込み
# ============================================================
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["開催日", "レース", "着", "艇", "登番", "モーター", "展示", "進入", "スタート"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} に必要な列がありません: {missing}")

    for col in required[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["着"].isin([1, 2, 3, 4, 5, 6])].copy()
    df = df.dropna(subset=FEATURES).copy()

    df["着"] = df["着"].astype(int)
    df["艇"] = df["艇"].astype(int)
    df["レース"] = pd.to_numeric(df["レース"], errors="coerce").astype(int)

    df["race_key"] = df["開催日"].astype(str) + "_R" + df["レース"].astype(str)

    # 1号艇を除いて5艇で学習
    df = df[df["艇"] != 1].copy()

    df["y_2nd"] = (df["着"] == 2).astype(int)
    df["y_3rd"] = (df["着"] == 3).astype(int)

    return df.reset_index(drop=True)


# ============================================================
# 学習
# ============================================================
def train_binary_model(train_df, test_df, target_col, model_name):
    # trainをさらにtrain/validに分割（時系列なのでシャッフルしない）
    split_idx = int(len(train_df) * 0.8)
    tr = train_df.iloc[:split_idx]
    va = train_df.iloc[split_idx:]

    X_tr, y_tr = tr[FEATURES], tr[target_col]
    X_va, y_va = va[FEATURES], va[target_col]
    X_test = test_df[FEATURES]

    train_ds = lgb.Dataset(X_tr, label=y_tr)
    valid_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
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
        valid_sets=[valid_ds],   # ← validのみ監視、testは触らない
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


# ============================================================
# 候補作成
# ============================================================
def actual_sanrentan(race_df):
    second_row = race_df[race_df["着"] == 2]
    third_row = race_df[race_df["着"] == 3]

    if len(second_row) != 1 or len(third_row) != 1:
        return None

    second = int(second_row.iloc[0]["艇"])
    third = int(third_row.iloc[0]["艇"])

    return f"1-{second}-{third}"


def make_top5_candidates(race_df, model_2nd, model_3rd, top_n=5):
    race_df = race_df.copy()

    race_df["p2"] = model_2nd.predict(race_df[FEATURES])
    race_df["p3"] = model_3rd.predict(race_df[FEATURES])

    candidates = []
    boats = race_df["艇"].astype(int).tolist()

    p2_map = dict(zip(race_df["艇"].astype(int), race_df["p2"]))
    p3_map = dict(zip(race_df["艇"].astype(int), race_df["p3"]))

    for second in boats:
        for third in boats:
            if second == third:
                continue
            score = p2_map[second] * p3_map[third]
            candidates.append((f"1-{second}-{third}", score))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return [x[0] for x in candidates[:top_n]]


# ============================================================
# 評価
# ============================================================
def evaluate(test_df, model_2nd, model_3rd, top_n=5):
    results = []

    for race_key, grp in test_df.groupby("race_key", sort=False):
        actual = actual_sanrentan(grp)
        if actual is None:
            continue

        preds = make_top5_candidates(grp, model_2nd, model_3rd, top_n=top_n)
        hit = int(actual in preds)

        row = {
            "race_key": race_key,
            "実際": actual,
            "的中": hit,
        }
        for i, p in enumerate(preds, start=1):
            row[f"予測{i}"] = p

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# メイン
# ============================================================
def main():
    print("CSVからデータ読み込み中...")
    train = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)

    print(f"train: {len(train):,} 行 / {train['race_key'].nunique():,} レース")
    print(f"test : {len(test):,} 行 / {test['race_key'].nunique():,} レース")

    model_2nd = train_binary_model(train, test, "y_2nd", "2着モデル")
    model_3rd = train_binary_model(train, test, "y_3rd", "3着モデル")

    df_result = evaluate(test, model_2nd, model_3rd, top_n=TOP_N)

    # 評価関数を呼ぶ前に追加
    df_train_result = evaluate(train, model_2nd, model_3rd, top_n=TOP_N)
    train_hit_rate = df_train_result["的中"].mean()
    print(f"train的中率: {train_hit_rate:.4f} ({train_hit_rate*100:.2f}%)")

    if len(df_result) == 0:
        print("評価対象レースがありません。")
        return

    hit_rate = df_result["的中"].mean()
    n_hits = int(df_result["的中"].sum())
    n_races = len(df_result)

    print(f"\n{'='*50}")
    print("1号艇1着側 Top5結果")
    print(f"予測レース数 : {n_races:,}")
    print(f"的中数       : {n_hits:,}")
    print(f"的中率       : {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"{'='*50}")

    df_result.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n結果保存完了: {RESULT_PATH}")


if __name__ == "__main__":
    main()