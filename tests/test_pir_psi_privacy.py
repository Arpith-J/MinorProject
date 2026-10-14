"""Statistical privacy tests for PIR-PSI (Sub-phase 8e).

Lemma 2 of Erhili 2024 states that, for every server ``n``, the distribution
of the query :math:`Q_n` is *independent* of the demand ``W`` and the
side-information set ``S``.  These tests verify that property empirically by
sampling queries under different ``(W, S)`` and comparing their empirical
distributions.

Because the support of a query vector lives in a large space, the tests
compare two well-chosen statistics that are invariant under the coordinate
relabeling that maps one ``(W, S)`` to another:

* the **support size** (number of non-zero entries), and
* the **canonical vector** (the sorted tuple of entries).

If the full distributions were to differ, at least one of these projections
would differ; the support-size projection is especially sensitive because the
non-uniform distribution is defined purely through support sizes.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import pytest

from src.pir_psi.query import build_query

# Upper-tail critical values of the chi-square distribution at alpha = 0.01.
CHI2_CRITICAL_001 = {
    1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086,
    6: 16.812, 7: 18.475, 8: 20.090, 9: 21.666, 10: 23.209,
    11: 24.725, 12: 26.217, 13: 27.688, 14: 29.141, 15: 30.578,
    16: 32.000, 17: 33.409, 18: 34.805, 19: 36.191, 20: 37.566,
}

N, K, M = 4, 5, 2
TRIALS = 10_000


def _collect(
    W: int, S: Sequence[int], trials: int, seed: int, server: int = 0
) -> List[Tuple[int, ...]]:
    """Sample ``trials`` query vectors for server ``server`` under ``(W, S)``."""
    rng = random.Random(seed)
    out: List[Tuple[int, ...]] = []
    for _ in range(trials):
        plan = build_query(W, list(S), K, N, M, rng)
        out.append(tuple(int(x) for x in plan.vectors[server]))
    return out


def _support_hist(vectors: Sequence[Tuple[int, ...]]) -> Counter:
    return Counter(sum(1 for x in v if x != 0) for v in vectors)


def _canonical_hist(vectors: Sequence[Tuple[int, ...]]) -> Counter:
    return Counter(tuple(sorted(v)) for v in vectors)


def _tv(hist_a: Counter, hist_b: Counter) -> float:
    """Total variation distance between two count histograms."""
    total_a = sum(hist_a.values())
    total_b = sum(hist_b.values())
    keys = set(hist_a) | set(hist_b)
    return 0.5 * sum(
        abs(hist_a.get(k, 0) / total_a - hist_b.get(k, 0) / total_b) for k in keys
    )


def _chi_square_p_over_001(hist_a: Counter, hist_b: Counter) -> Tuple[float, int]:
    """Return the chi-square statistic and its degrees of freedom.

    Categories are merged greedily until every pooled expected count is at
    least five, keeping the chi-square approximation valid.
    """
    total = sum(hist_a.values()) + sum(hist_b.values())
    keys = sorted(set(hist_a) | set(hist_b))
    counts = [
        (key, hist_a.get(key, 0), hist_b.get(key, 0))
        for key in keys
    ]
    # Merge rare categories into a single bucket.
    merged_obs_a: List[float] = []
    merged_obs_b: List[float] = []
    rare_a = rare_b = 0
    for _, a, b in counts:
        expected = (a + b) / 2
        if expected < 5:
            rare_a += a
            rare_b += b
        else:
            merged_obs_a.append(a)
            merged_obs_b.append(b)
    if rare_a + rare_b > 0:
        merged_obs_a.append(rare_a)
        merged_obs_b.append(rare_b)
    statistic = 0.0
    for a, b in zip(merged_obs_a, merged_obs_b):
        expected = (a + b) / 2
        if expected > 0:
            statistic += (a - expected) ** 2 / expected
            statistic += (b - expected) ** 2 / expected
    df = max(1, len(merged_obs_a) - 1)
    return statistic, df


# (W, S) pairs exercising the three scenarios of the spec.
PAIRS = [
    # Two different demands, same side information.
    ((2, (0, 1)), (3, (0, 1))),
    # Same demand, two different side-information sets.
    ((2, (0, 1)), (2, (0, 3))),
    # Two different (W, S) pairs.
    ((2, (0, 1)), (4, (2, 3))),
]


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("server", [0, 2])
def test_support_size_tv_below_threshold(pair, server):
    (w1, s1), (w2, s2) = pair
    h1 = _support_hist(_collect(w1, s1, TRIALS, 100 + server, server))
    h2 = _support_hist(_collect(w2, s2, TRIALS, 200 + server, server))
    assert _tv(h1, h2) < 0.05


@pytest.mark.parametrize("pair", PAIRS)
def test_support_size_chi_square(pair):
    (w1, s1), (w2, s2) = pair
    h1 = _support_hist(_collect(w1, s1, TRIALS, 11))
    h2 = _support_hist(_collect(w2, s2, TRIALS, 22))
    statistic, df = _chi_square_p_over_001(h1, h2)
    assert statistic < CHI2_CRITICAL_001[df], (
        f"chi-square {statistic:.3f} exceeds critical value "
        f"{CHI2_CRITICAL_001[df]} (df={df}); p <= 0.01"
    )


@pytest.mark.parametrize("pair", PAIRS)
def test_canonical_vector_tv_below_threshold(pair):
    (w1, s1), (w2, s2) = pair
    h1 = _canonical_hist(_collect(w1, s1, TRIALS, 31))
    h2 = _canonical_hist(_collect(w2, s2, TRIALS, 32))
    assert _tv(h1, h2) < 0.05


@pytest.mark.parametrize("pair", PAIRS)
def test_full_distribution_permutation_test(pair):
    """Distribution-free check on the canonical-vector distribution.

    Pools the two samples, randomly re-splits them many times and compares
    the observed total-variation distance against the permutation null.
    """
    (w1, s1), (w2, s2) = pair
    sample = _collect(w1, s1, 2000, 41) + _collect(w2, s2, 2000, 42)
    n = len(sample)
    observed = _tv(
        _canonical_hist(sample[: n // 2]), _canonical_hist(sample[n // 2:])
    )
    rng = random.Random(7)
    special = 0
    permutations = 200
    for _ in range(permutations):
        rng.shuffle(sample)
        stat = _tv(
            _canonical_hist(sample[: n // 2]), _canonical_hist(sample[n // 2:])
        )
        if stat >= observed:
            special += 1
    p_value = (special + 1) / (permutations + 1)
    assert p_value > 0.01


def test_m_zero_query_distribution_is_demand_invariant():
    """For M=0 the demand still cannot be identified from a server's view."""
    N0, K0, M0 = 3, 6, 0
    rng = random.Random(0)

    def support_hist(w):
        hist: Counter = Counter()
        local = random.Random(99)
        for _ in range(TRIALS):
            plan = build_query(w, [], K0, N0, M0, local)
            hist[sum(1 for x in plan.vectors[0] if x != 0)] += 1
        return hist

    h1 = support_hist(1)
    h2 = support_hist(4)
    assert _tv(h1, h2) < 0.05
