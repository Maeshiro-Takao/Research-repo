"""
丸亀学習用_レースデータ.csv と 丸亀学習用_選手データ.csv を併用して
LightGBM LambdaRank 着順モデルを学習し、3連単（1-2-3着の順番）予測を行う。

レース単位のランク学習で各艇のスコアを予測し、
Plackett-Luce モデルで120通りの3連単確率を算出する。

使い方:
    python train_trifecta.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent
RACE_DATA_PATH = BASE_DIR / "編集データ" / "丸亀学習用_レースデータ.csv"
PLAYER_DATA_PATH = BASE_DIR / "編集データ" / "丸亀学習用_選手データ.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RACE_KEY = ["開催日", "日目", "レース"]
RACE_ROW_KEY = ["開催日", "日目", "レース", "艇"]

RACE_EXCLUDE_COLUMNS = {"着", "選手名", "日目", "開催日"}
PLAYER_META_COLS = ["名前漢字", "算出期間自", "算出期間至"]
RACE_BASE_FEATURES = [
    "艇", "登番", "モーター", "ボート", "展示",
    "展示順位", "展示差", "レース", "風速", "波高", "天気", "風向",
]
TOP_N_LIST = [1, 3, 5, 10, 30]
SHAP_SAMPLE_SIZE = 2000

FEATURES: list[str] = []
CATEGORICAL = ["艇", "登番", "モーター", "ボート", "天気", "風向", "級"]
FEATURE_LABELS: dict[str, str] = {}

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "verbosity": -1,
    "seed": 42,
    "num_leaves": 126,
    "max_depth": 6,
    "learning_rate": 0.02834061120740455,
    "min_data_in_leaf": 123,
    "feature_fraction": 0.6598699690730964,
    "bagging_fraction": 0.6015757233072625,
    "bagging_freq": 1,
    "lambda_l1": 0.2752764624363988,
    "lambda_l2": 5.8874346025965005,
    "num_boost_round": 215
}
NUM_BOOST_ROUND = 235


def configure_features(player_df: pd.DataFrame) -> None:
    """レースデータ・選手データの全特徴量を設定する"""
    global FEATURES, FEATURE_LABELS

    player_features = [
        c for c in player_df.columns
        if c not in PLAYER_META_COLS and c != "登番"
    ]
    FEATURES = RACE_BASE_FEATURES + player_features
    FEATURE_LABELS = {
        "展示": "展示タイム",
        "レース": "レース番号",
        "平均スタートタイミング": "平均ST",
        **{f"{c}コース平均スタートタイミング": f"{c}コース平均ST" for c in range(1, 7)},
        **{c: c for c in FEATURES},
    }


JAPANESE_FONT_CANDIDATES = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Hiragino Maru Gothic ProN",
    "Yu Gothic",
    "YuGothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAGothic",
    "MS Gothic",
]

def setup_japanese_font() -> str | None:
    """利用可能な日本語フォントを設定する"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in JAPANESE_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name

    for font in font_manager.fontManager.ttflist:
        if any(k in font.name for k in ("Hiragino", "Noto Sans CJK", "Yu Gothic", "Meiryo")):
            plt.rcParams["font.family"] = font.name
            plt.rcParams["axes.unicode_minus"] = False
            return font.name

    plt.rcParams["axes.unicode_minus"] = False
    return None


def merge_player_data(race_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """登番と開催日（算出期間）で選手データを結合する"""
    race = race_df.copy()
    player = player_df.copy()
    race["開催日"] = pd.to_datetime(race["開催日"])

    if {"算出期間自", "算出期間至"}.issubset(player.columns):
        player["算出期間自"] = pd.to_datetime(player["算出期間自"])
        player["算出期間至"] = pd.to_datetime(player["算出期間至"])
        player_cols = [c for c in player.columns if c != "登番"]
        merged = race.merge(player, on="登番", how="left")
        period_match = (
            merged["算出期間自"].notna()
            & (merged["開催日"] >= merged["算出期間自"])
            & (merged["開催日"] <= merged["算出期間至"])
        )
        matched = (
            merged.loc[period_match, RACE_ROW_KEY + player_cols]
            .drop_duplicates(RACE_ROW_KEY)
        )
        out = race.merge(matched, on=RACE_ROW_KEY, how="left")
    else:
        player = player.drop_duplicates(subset=["登番"], keep="last")
        player_cols = [c for c in player.columns if c != "登番"]
        out = race.merge(player, on="登番", how="left")

    return out.drop(columns=[c for c in PLAYER_META_COLS if c in out.columns])


def load_data() -> pd.DataFrame:
    print(f"データ読み込み: {RACE_DATA_PATH.name}, {PLAYER_DATA_PATH.name}")
    race_df = pd.read_csv(RACE_DATA_PATH)
    player_df = pd.read_csv(PLAYER_DATA_PATH)
    configure_features(player_df)
    df = merge_player_data(race_df, player_df)

    matched = df["級"].notna().sum() if "級" in df.columns else 0
    print(f"  全体: {len(df)} 行 / {len(df) // 6} レース")
    print(f"  選手データ結合: {matched} 行 ({matched / len(df):.1%})")
    print(f"  特徴量数: {len(FEATURES)}")
    return df


def filter_complete_races(df: pd.DataFrame) -> pd.DataFrame:
    """6艇そろったレースのみを残す"""
    sizes = df.groupby(RACE_KEY, sort=False)["艇"].transform("size")
    return df.loc[sizes == 6].copy()


def prepare(df: pd.DataFrame, encoders=None):
    df = filter_complete_races(df)
    df = df.sort_values(RACE_KEY).reset_index(drop=True)
    boat_numbers = df["艇"].astype(int).values
    df["展示順位"] = df.groupby(RACE_KEY)["展示"].rank(method="min")
    df["展示差"] = df["展示"] - df.groupby(RACE_KEY)["展示"].transform("mean")
    groups = df.groupby(RACE_KEY, sort=False).size().tolist()
    df = df.drop(columns=["日目", "選手名"], errors="ignore")
    encoders = encoders or {}
    for col in CATEGORICAL:
        if col not in encoders:
            encoders[col] = {v: i for i, v in enumerate(df[col].astype(str).unique())}
        df[col] = df[col].astype(str).map(encoders[col]).fillna(-1).astype(int)

    x = df[FEATURES]
    y = 7 - df["着"].astype(int)
    return x, y, groups, encoders, boat_numbers


def get_actual_trifecta(race_df: pd.DataFrame) -> tuple[int, int, int]:
    top3 = race_df.sort_values("着").head(3)
    return tuple(top3["艇"].astype(int).tolist())  # type: ignore[return-value]


def calc_trifecta_probs_from_scores(scores: np.ndarray) -> dict[tuple[int, int, int], float]:
    """ランク学習スコアから Plackett-Luce モデルで3連単確率を算出する"""
    n = len(scores)
    exp_scores = np.exp(scores - scores.max())
    results: dict[tuple[int, int, int], float] = {}

    for i in range(n):
        p1 = max(exp_scores[i] / exp_scores.sum(), 1e-12)
        rem1 = [b for b in range(n) if b != i]
        sum_rem1 = exp_scores[rem1].sum()

        for j in rem1:
            p2 = max(exp_scores[j] / sum_rem1, 1e-12)
            rem2 = [b for b in rem1 if b != j]
            sum_rem2 = exp_scores[rem2].sum()

            for k in rem2:
                p3 = max(exp_scores[k] / sum_rem2, 1e-12)
                results[(i + 1, j + 1, k + 1)] = p1 * p2 * p3

    return results


def run_shap_analysis(model: lgb.Booster, x_sample: pd.DataFrame):
    """ランク学習モデルの SHAP 分析"""
    x_display = x_sample.rename(columns={c: FEATURE_LABELS.get(c, c) for c in x_sample.columns})
    shap_values = shap.TreeExplainer(model).shap_values(x_sample)

    bar_path = MODEL_DIR / "trifecta_shap_importance.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_display, plot_type="bar", show=False)
    plt.title("特徴量の重要度（着順スコアへの影響）", fontsize=14)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    beeswarm_path = MODEL_DIR / "trifecta_shap_beeswarm.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, x_display, show=False)
    plt.title("特徴量の影響方向（赤=有利, 青=不利）", fontsize=14)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame({
        "特徴量": x_display.columns,
        "重要度": np.abs(shap_values).mean(axis=0),
    }).sort_values("重要度", ascending=False)
    csv_path = MODEL_DIR / "trifecta_shap_importance.csv"
    importance.to_csv(csv_path, index=False, encoding="UTF-8-sig")

    print(f"  重要度グラフ: {bar_path}")
    print(f"  蜂群プロット: {beeswarm_path}")
    print(f"  重要度CSV: {csv_path}")

    print("\n=== 特徴量重要度 TOP5 ===")
    for _, row in importance.head(5).iterrows():
        print(f"    {row['特徴量']}: {row['重要度']:.4f}")


def predict_race_trifecta(
    model: lgb.Booster, race_df: pd.DataFrame, encoders: dict
) -> tuple[tuple[int, int, int], list[tuple[tuple[int, int, int], float]]]:
    """1レース分の3連単予測（最上位と確率順ランキング）"""
    x, _, _, _, boat_numbers = prepare(race_df, encoders)
    scores = model.predict(x)
    score_arr = np.zeros(6)
    for idx, boat in enumerate(boat_numbers):
        score_arr[boat - 1] = scores[idx]

    trifecta_probs = calc_trifecta_probs_from_scores(score_arr)
    ranking = sorted(trifecta_probs.items(), key=lambda x: x[1], reverse=True)
    return ranking[0][0], ranking


def evaluate_trifecta(
    model: lgb.Booster, df: pd.DataFrame, encoders: dict, label: str
) -> dict:
    hits = {n: 0 for n in TOP_N_LIST}
    first_hits = 0
    second_hits = 0
    third_hits = 0
    logloss_sum = 0.0
    n_races = 0

    for _, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue

        actual = get_actual_trifecta(race_df)
        pred_top, ranking = predict_race_trifecta(model, race_df, encoders)
        prob_map = dict(ranking)

        for n in TOP_N_LIST:
            top_n = [combo for combo, _ in ranking[:n]]
            if actual in top_n:
                hits[n] += 1

        if pred_top[0] == actual[0]:
            first_hits += 1
        if pred_top[:2] == actual[:2]:
            second_hits += 1
        if pred_top == actual:
            pass  # counted in hits[1]

        true_prob = max(prob_map.get(actual, 1e-12), 1e-12)
        logloss_sum += -np.log(true_prob)
        n_races += 1

    print(f"\n=== 3連単評価 ({label}) ===")
    print(f"  レース数: {n_races}")
    for n in TOP_N_LIST:
        print(f"  3連単 TOP{n} 的中率: {hits[n] / n_races:.2%}")
    print(f"  1着艇番 的中率: {first_hits / n_races:.2%}")
    print(f"  2連単 的中率:   {second_hits / n_races:.2%}")
    print(f"  3連単 的中率:   {hits[1] / n_races:.2%}")
    print(f"  真の3連単 平均-log確率: {logloss_sum / n_races:.4f}")

    return {
        "n_races": n_races,
        "trifecta_top1": hits[1] / n_races,
        **{f"trifecta_top{n}": hits[n] / n_races for n in TOP_N_LIST},
        "first_hit": first_hits / n_races,
        "exacta_hit": second_hits / n_races,
    }


def save_trifecta_predictions(
    model: lgb.Booster, df: pd.DataFrame, encoders: dict, output_path: Path
):
    rows = []
    for key, race_df in df.groupby(RACE_KEY, sort=False):
        if len(race_df) != 6:
            continue
        actual = get_actual_trifecta(race_df)
        pred_top, ranking = predict_race_trifecta(model, race_df, encoders)
        top10 = ranking[:10]
        rows.append({
            "開催日": key[0],
            "レース": key[2],
            "実際3連単": "-".join(map(str, actual)),
            "予測3連単": "-".join(map(str, pred_top)),
            "予測確率": ranking[0][1],
            "的中": pred_top == actual,
            "TOP10": " / ".join(
                f"{'-'.join(map(str, combo))}({prob:.4f})" for combo, prob in top10
            ),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False, encoding="UTF-8-sig")
    print(f"  予測結果CSV: {output_path}")


def main():
    font = setup_japanese_font()
    if font:
        print(f"日本語フォント: {font}")
    else:
        print("警告: 日本語フォントが見つかりません。グラフの日本語が文字化けする可能性があります。")

    df = load_data()
    print(
        f"  期間: {pd.to_datetime(df['開催日']).min().date()} 〜 "
        f"{pd.to_datetime(df['開催日']).max().date()}"
    )

    x_train, y_train, groups, encoders, _ = prepare(df)
    train_set = lgb.Dataset(
        x_train, label=y_train, group=groups, categorical_feature=CATEGORICAL
    )

    print("\n着順モデル学習中（LambdaRank / 3連単予測の基礎モデル）...")
    model = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(50)],
    )

    evaluate_trifecta(model, df, encoders, "全体")

    model.save_model(str(MODEL_DIR / "lgbm_trifecta_model.txt"))
    joblib.dump(encoders, MODEL_DIR / "trifecta_encoders.pkl")
    print(f"\nモデル保存: {MODEL_DIR / 'lgbm_trifecta_model.txt'}")

    print("\n3連単予測を保存...")
    save_trifecta_predictions(
        model, df, encoders, MODEL_DIR / "trifecta_predictions.csv"
    )

    print("\nSHAP分析中...")
    x_sample = x_train.sample(min(SHAP_SAMPLE_SIZE, len(x_train)), random_state=42)
    run_shap_analysis(model, x_sample)


if __name__ == "__main__":
    main()
