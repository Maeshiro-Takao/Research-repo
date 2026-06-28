"""
Plackett–Luce モデル

LightGBM Ranker が出力した各艇のランキングスコアを強さパラメータ π_i = exp(s_i) として、
逐次選択確率 P(i|S) = π_i / Σ_{j∈S} π_j により順位確率・3連単確率を計算する。
単一 Softmax（120通りへの一括正規化）ではない。
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

ALL_TRIFECTA_COMBOS: list[tuple[int, int, int]] = [
    (i, j, k)
    for i in range(1, 7)
    for j in range(1, 7)
    if j != i
    for k in range(1, 7)
    if k not in (i, j)
]


def scores_to_strengths(scores: np.ndarray) -> np.ndarray:
    """ランキングスコアを Plackett–Luce 強さパラメータへ変換（数値安定化）"""
    scores = np.asarray(scores, dtype=float)
    return np.exp(scores - scores.max())


def plackett_luce_trifecta_probs(
    scores: np.ndarray,
    boats: list[int] | None = None,
) -> dict[tuple[int, int, int], float]:
    """
    6艇スコアから全120通りの3連単確率を Plackett–Luce で算出する。

    P(1着=i, 2着=j, 3着=k) =
        (π_i / Σπ) × (π_j / Σ_{≠i}π) × (π_k / Σ_{≠i,j}π)
    """
    strengths = scores_to_strengths(scores)
    n = len(strengths)
    if boats is None:
        boats = list(range(1, n + 1))
    if len(boats) != n:
        raise ValueError(f"boats の長さ ({len(boats)}) と scores ({n}) が一致しません")

    results: dict[tuple[int, int, int], float] = {}
    for i in range(n):
        p1 = strengths[i] / strengths.sum()
        rem1_idx = [b for b in range(n) if b != i]
        rem1_str = strengths[rem1_idx]
        sum_rem1 = rem1_str.sum()
        for ji, j in enumerate(rem1_idx):
            p2 = rem1_str[ji] / sum_rem1
            rem2_idx = [b for b in rem1_idx if b != j]
            rem2_str = strengths[rem2_idx]
            sum_rem2 = rem2_str.sum()
            for ki, k in enumerate(rem2_idx):
                p3 = rem2_str[ki] / sum_rem2
                results[(boats[i], boats[j], boats[k])] = p1 * p2 * p3
    return results


def plackett_luce_position_probs(scores: np.ndarray) -> np.ndarray:
    """
    各艇の各着順確率を返す。shape = (n_boats, n_boats)。
    result[i, k] = P(艇 i+1 が k+1 着)
    """
    strengths = scores_to_strengths(scores)
    n = len(strengths)
    probs = np.zeros((n, n))
    for perm in permutations(range(n)):
        p = 1.0
        remaining = list(strengths)
        remaining_idx = list(range(n))
        for boat_idx in perm:
            idx_in_rem = remaining_idx.index(boat_idx)
            p *= remaining[idx_in_rem] / sum(remaining)
            remaining.pop(idx_in_rem)
            remaining_idx.pop(idx_in_rem)
        for pos, boat_idx in enumerate(perm):
            probs[boat_idx, pos] += p
    return probs


def rank_trifecta_by_prob(
    prob_map: dict[tuple[int, int, int], float],
) -> list[tuple[tuple[int, int, int], float, int]]:
    """確率降順に (組み合わせ, 確率, 順位) のリストを返す"""
    ranked = sorted(prob_map.items(), key=lambda x: (-x[1], x[0]))
    return [(combo, prob, rank + 1) for rank, (combo, prob) in enumerate(ranked)]


def combo_to_str(combo: tuple[int, int, int]) -> str:
    return f"{combo[0]}-{combo[1]}-{combo[2]}"


def prob_sum(prob_map: dict[tuple[int, int, int], float]) -> float:
    return float(sum(prob_map.values()))


def compute_ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """1レース分の NDCG@k（relevance が大きいほど上位着順）"""
    order = np.argsort(-scores)
    rel = relevance[order][:k]
    dcg = sum((2.0 ** r - 1.0) / np.log2(i + 2) for i, r in enumerate(rel))
    ideal_order = np.argsort(-relevance)
    ideal_rel = relevance[ideal_order][:k]
    idcg = sum((2.0 ** r - 1.0) / np.log2(i + 2) for i, r in enumerate(ideal_rel))
    return float(dcg / idcg) if idcg > 0 else 0.0


def predict_trifecta_from_scores(
    scores: np.ndarray,
    boats: list[int] | None = None,
) -> tuple[dict[tuple[int, int, int], float], list[tuple[tuple[int, int, int], float, int]]]:
    """スコア → 120通り確率マップ + 確率順ランキング"""
    prob_map = plackett_luce_trifecta_probs(scores, boats=boats)
    ranking = rank_trifecta_by_prob(prob_map)
    return prob_map, ranking
