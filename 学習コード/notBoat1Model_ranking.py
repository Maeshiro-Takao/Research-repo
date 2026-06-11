import math
from itertools import permutations
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# ============================================================
# 設定
# ============================================================
BASE_DIR = Path("/Users/maeshirotakao/ボートデータ/丸亀データ")

TRAIN_PATH = BASE_DIR / "丸亀データ_学習用_選手付き_1号艇以外1着.csv"
TEST_PATH  = BASE_DIR / "丸亀データ_テスト用_選手付き_1号艇以外1着.csv"
RESULT_PATH = BASE_DIR / "結果_1号艇以外1着_ranking_top5.csv"

FEATURE_COLS = [
    "艇", "登番", "モーター", "展示", "スタート",
    "勝率", "複勝率", "1着回数", "2着回数", "出走回数",
    "優出回数", "優勝回数", "平均ST",
    "1コース進入回数", "1コース複勝率", "1コースST", "1コースST順位",
    "2コース進入回数", "2コース複勝率", "2コースST", "2コースST順位",
    "3コース進入回数", "3コース複勝率", "3コースST", "3コースST順位",
    "4コース進入回数", "4コース複勝率", "4コースST", "4コースST順位",
    "5コース進入回数", "5コース複勝率", "5コースST", "5コースST順位",
    "6コース進入回数", "6コース複勝率", "6コースST", "6コースST順位",
]
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

    numeric_cols = ["着", "艇", "登番"] + FEATURE_COLS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["着"].isin([1, 2, 3, 4, 5, 6])].copy()
    df = df.dropna(subset=FEATURE_COLS).copy()

    df["着"] = df["着"].astype(int)
    df["艇"] = df["艇"].astype(int)
    df["レース"] = pd.to_numeric(df["レース"], errors="coerce").astype(int)

    df["race_key"] = df["開催日"].astype(str) + "_R" + df["レース"].astype(str)

    # 6艇そろっているレースだけ
    race_sizes = df.groupby("race_key").size()
    valid_races = race_sizes[race_sizes == 6].index
    df = df[df["race_key"].isin(valid_races)].copy()

    # 艇が1〜6そろっているレースだけ
    valid_keys = []
    for race_key, grp in df.groupby("race_key", sort=False):
        boats = sorted(grp["艇"].astype(int).tolist())
        if boats == [1, 2, 3, 4, 5, 6]:
            valid_keys.append(race_key)

    df = df[df["race_key"].isin(valid_keys)].copy()

    # 1号艇が1着ではないレースだけ念のため再確認
    first_boat = (
        df.loc[df["着"] == 1, ["race_key", "艇"]]
        .rename(columns={"艇": "1着艇"})
        .drop_duplicates("race_key")
    )
    target_races = first_boat[first_boat["1着艇"] != 1]["race_key"]
    df = df[df["race_key"].isin(target_races)].copy()

    return df.reset_index(drop=True)


# ============================================================
# ranking用ラベル作成
# ============================================================
def add_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """
    ranking 学習用の relevance を作る
    1着 = 3
    2着 = 2
    3着 = 1
    4〜6着 = 0
    """
    df = df.copy()
    df["rel"] = 0
    df.loc[df["着"] == 1, "rel"] = 3
    df.loc[df["着"] == 2, "rel"] = 2
    df.loc[df["着"] == 3, "rel"] = 1
    return df


# ============================================================
# 学習
# ============================================================
def train_rank_model(train_df: pd.DataFrame, test_df: pd.DataFrame, model_name: str):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["rel"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["rel"]

    group_train = train_df.groupby("race_key", sort=False).size().tolist()
    group_test = test_df.groupby("race_key", sort=False).size().tolist()

    train_ds = lgb.Dataset(X_train, label=y_train, group=group_train)
    valid_ds = lgb.Dataset(X_test, label=y_test, group=group_test, reference=train_ds)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.07,
        "feature_fraction": 0.4,
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
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


# ============================================================
# 3連単候補生成
# ============================================================
def actual_sanrentan(race_df: pd.DataFrame) -> str | None:
    top3 = race_df.sort_values("着").head(3)["艇"].astype(int).tolist()
    if len(top3) != 3:
        return None
    return f"{top3[0]}-{top3[1]}-{top3[2]}"


def predict_top5_for_race(race_df: pd.DataFrame, model, top_n: int = 5):
    grp = race_df.copy().sort_values("艇")
    grp["score"] = model.predict(grp[FEATURE_COLS])

    score_map = dict(zip(grp["艇"].astype(int), grp["score"]))
    boats = grp["艇"].astype(int).tolist()

    candidates = []
    for a, b, c in permutations(boats, 3):
        # このデータは 1号艇が1着ではないレース専用
        if a == 1:
            continue

        # 積より log 和の方が安定
        score = (
            math.log(max(score_map[a], 1e-15)) +
            math.log(max(score_map[b], 1e-15)) +
            math.log(max(score_map[c], 1e-15))
        )
        candidates.append((f"{a}-{b}-{c}", score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in candidates[:top_n]]


# ============================================================
# 評価
# ============================================================
def evaluate(test_df: pd.DataFrame, model, top_n: int = 5) -> pd.DataFrame:
    results = []

    for race_key, grp in test_df.groupby("race_key", sort=False):
        actual = actual_sanrentan(grp)
        if actual is None:
            continue

        preds = predict_top5_for_race(grp, model, top_n=top_n)
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

    train = add_relevance(train)
    test = add_relevance(test)

    print(f"train: {train['race_key'].nunique():,} レース / {len(train):,} 行")
    print(f"test : {test['race_key'].nunique():,} レース / {len(test):,} 行")

    model = train_rank_model(train, test, "rankingモデル")

    df_result = evaluate(test, model, top_n=TOP_N)

    if len(df_result) == 0:
        print("評価対象レースがありません。")
        return

    hit_rate = df_result["的中"].mean()
    n_hits = int(df_result["的中"].sum())
    n_races = len(df_result)

    print(f"\n{'='*50}")
    print("1号艇1着以外側 Ranking Top5結果")
    print(f"予測レース数 : {n_races:,}")
    print(f"的中数       : {n_hits:,}")
    print(f"的中率       : {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"{'='*50}")

    df_result.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n結果保存完了: {RESULT_PATH}")
    print(df_result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()