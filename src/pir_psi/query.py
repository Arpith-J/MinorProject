"""Randomized query generation for the PIR-PSI scheme.

This module implements the query-generation algorithm of Erhili &
Heidarzadeh (2024), "Achieving Capacity of PIR with Private Side Information
with Low Sub-packetization and without MDS Codes", Section IV (Algorithm 1).

Every message is treated as :math:`N-1` sub-packets, indexed
``0 .. N-1`` where index ``0`` denotes the *degenerate* all-zero sub-packet.
A query vector :math:`\\mathrm{v}_n \\in [0:N-1]^K` tells server ``n`` which
sub-packet of each message to add into its answer.

The queries are drawn from the *non-uniform* probability distribution of
Lemma 1 of the paper.  The distribution depends only on the query support
size, which is exactly what makes each server's marginal view independent of
the demand :math:`W` and the side-information set :math:`S` (privacy).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import comb
from typing import Dict, List, Optional, Sequence

import numpy as np

# Re-exported in the package ``__init__`` for convenience.
__all__ = [
    "QueryPlan",
    "combinatorial_m",
    "probability_matrix",
    "sample_query_parameters",
    "generate_query_vectors",
    "permute_vectors",
    "build_query",
]


def _binom(n: int, k: int) -> int:
    """Generalized binomial coefficient with the paper's conventions.

    ``C(n, k) = 0`` when ``k < 0`` or ``k > n`` for ``n >= 0``.  The
    convention ``C(-1, 0) = 1`` is required so that the ``M = 0`` case
    (which the paper does not cover explicitly) reduces to the classical
    PIR distribution.

    Args:
        n: Upper index (may be negative here, unlike :func:`math.comb`).
        k: Lower index.

    Returns:
        The generalized binomial coefficient as a Python ``int``.
    """
    if k < 0:
        return 0
    if n < 0:
        return 1 if k == 0 else 0
    if k > n:
        return 0
    return comb(n, k)


def combinatorial_m(i: int, j: int, M: int, N: int) -> int:
    """Compute the coefficient :math:`m_{i,j}` of Erhili 2024, eq. (5).

    The coefficient makes the non-uniform distribution of Lemma 1 a valid
    probability distribution.  It is defined by

    .. math::
        m_{i,j} = \\sum_{k=0}^{i+j-M-1} (-1)^k \\binom{M+k-1}{k}
                  (N-1)^{i+j-M-k},

    with the special cases :math:`m_{0,0} = 1` and :math:`m_{i,j} = 0` for
    :math:`1 \\le i+j \\le M`.

    Args:
        i: Size of the side-information subset used (``0 <= i <= M``).
        j: Size of the interference subset used.
        M: Number of side-information messages.
        N: Number of servers.

    Returns:
        The integer coefficient :math:`m_{i,j}`.
    """
    if i == 0 and j == 0:
        return 1
    if 1 <= i + j <= M:
        return 0
    s = i + j
    total = 0
    for k in range(0, s - M):
        total += (
            (-1) ** k
            * _binom(M + k - 1, k)
            * (N - 1) ** (s - M - k)
        )
    return total


def probability_matrix(M: int, K: int, N: int) -> np.ndarray:
    """Return the ``(M+1) x (K-M)`` probability matrix :math:`P_{i,j}`.

    Implements Erhili 2024, eq. (4)::

        P[i][j] = C(M, i) * C(K-M-1, j) * m_{i,j} / N^{K-M-1}

    Args:
        M: Number of side-information messages.
        K: Total number of messages.
        N: Number of servers.

    Returns:
        A ``float`` ``np.ndarray`` of shape ``(M + 1, K - M)`` summing to
        one.

    Raises:
        ValueError: If the parameters do not satisfy the scheme's
            requirements (``N >= M + 1`` and ``K > M``).
    """
    if N < M + 1:
        raise ValueError(f"need N >= M+1, got N={N}, M={M}")
    if K <= M:
        raise ValueError(f"need K > M, got K={K}, M={M}")
    denom = N ** (K - M - 1)
    P = np.zeros((M + 1, K - M), dtype=float)
    for i in range(M + 1):
        for j in range(K - M):
            P[i, j] = (
                _binom(M, i)
                * _binom(K - M - 1, j)
                * combinatorial_m(i, j, M, N)
                / denom
            )
    if abs(float(P.sum()) - 1.0) > 1e-9:  # pragma: no cover - defensive
        raise AssertionError(f"probability matrix does not sum to 1: {P.sum()}")
    return P


def sample_query_parameters(
    M: int, K: int, N: int, rng: random.Random
) -> "tuple[int, int]":
    """Sample ``(I, J)`` from the non-uniform distribution :math:`P_{i,j}`.

    Args:
        M: Number of side-information messages.
        K: Total number of messages.
        N: Number of servers.
        rng: Seeded ``random.Random`` instance.

    Returns:
        A pair ``(I, J)`` with ``0 <= I <= M`` (number of side-information
        sub-packets used) and ``0 <= J <= K-M-1`` (number of interference
        sub-packets used).
    """
    P = probability_matrix(M, K, N)
    flat = P.ravel()
    idx = rng.choices(range(flat.size), weights=flat.tolist(), k=1)[0]
    width = K - M
    return idx // width, idx % width


def permute_vectors(
    vectors: Sequence[np.ndarray], rng: random.Random
) -> List[np.ndarray]:
    """Apply a uniformly random permutation ``sigma`` of ``[N]``.

    Given vectors :math:`u_1,\\dots,u_N` this returns
    :math:`v_n = u_{\\sigma(n)}`, matching the paper's step 6.

    Args:
        vectors: The ``N`` vectors to permute.
        rng: Seeded ``random.Random`` instance.

    Returns:
        A new list of ``N`` arrays with ``out[n] = vectors[sigma[n]]``.
    """
    n = len(vectors)
    sigma = rng.sample(range(n), n)
    return [np.array(vectors[sigma[k]], copy=True) for k in range(n)]


@dataclass
class QueryPlan:
    """The full query realization for one PIR-PSI retrieval.

    The client keeps every field so that it can cancel the side-information
    contributions during decoding; only :attr:`vectors` is ever sent to the
    servers.

    Attributes:
        vectors: The ``N`` query vectors sent to the servers.
        demand: The demanded message id ``W``.
        side_info_ids: The side-information id set ``S``.
        K: Number of messages.
        N: Number of servers.
        M: Number of side-information messages.
        I: Number of side-information sub-packets used.
        J: Number of interference sub-packets used.
        theta: The Bernoulli selector ``theta`` in ``{0, 1}``.
        pi: A permutation of sub-packet indices ``1 .. N-1`` (``pi[p]`` is
            the demand sub-packet recovered from ``u_{p+1}``).
        sigma: The server permutation, ``v[n] = u[sigma[n]]``.
        b: Mapping from side-information id to its chosen nonzero value.
        R: The ``I``-subset of side information used (a list of ids).
        excluded: For each ``p < I`` the element removed from ``R`` to form
            ``R_p`` (used when ``theta == 1``).
        c: Mapping from interference id to its chosen nonzero value.
    """

    vectors: List[np.ndarray]
    demand: int
    side_info_ids: List[int]
    K: int
    N: int
    M: int
    I: int
    J: int
    theta: int
    pi: List[int]
    sigma: List[int]
    b: Dict[int, int] = field(default_factory=dict)
    R: List[int] = field(default_factory=list)
    excluded: List[int] = field(default_factory=list)
    c: Dict[int, int] = field(default_factory=dict)

    def support_size(self, n: int) -> int:
        """Number of non-zero entries of the vector sent to server ``n``."""
        return int(np.count_nonzero(self.vectors[n]))


def build_query(
    msg_id: int,
    side_info_ids: Sequence[int],
    K: int,
    N: int,
    M: int,
    rng: random.Random,
) -> QueryPlan:
    """Construct a full query realization (Erhili 2024, Algorithm 1).

    Args:
        msg_id: The demanded message id ``W``.
        side_info_ids: The ``M`` side-information message ids ``S``.
        K: Total number of messages.
        N: Number of servers.
        M: Number of side-information messages.
        rng: Seeded ``random.Random`` instance.

    Returns:
        A :class:`QueryPlan` holding the ``N`` query vectors and all the
        randomness needed to decode.

    Raises:
        ValueError: If parameters are inconsistent (``N < M+1``,
            ``K <= M``, out-of-range ids, or ``M != len(side_info_ids)``).
    """
    if N < 2:
        raise ValueError(f"need at least 2 servers, got N={N}")
    if N < M + 1:
        raise ValueError(f"PIR-PSI requires N >= M+1, got N={N}, M={M}")
    if K <= M:
        raise ValueError(f"need K > M, got K={K}, M={M}")
    if not 0 <= msg_id < K:
        raise ValueError(f"msg_id {msg_id} out of range for K={K}")
    S = [int(s) for s in side_info_ids]
    if len(S) != M:
        raise ValueError(f"expected {M} side-info ids, got {len(S)}")
    if msg_id in S:
        raise ValueError(f"msg_id {msg_id} appears in side information")
    if len(set(S)) != len(S):
        raise ValueError("side information contains duplicate ids")
    for s in S:
        if not 0 <= s < K:
            raise ValueError(f"side info id {s} out of range for K={K}")

    # Step 1: a_1 .. a_{N-1}: place one distinct demand sub-packet each.
    pi = rng.sample(range(1, N), N - 1)  # permutation of 1 .. N-1
    a = [np.zeros(K, dtype=np.int64) for _ in range(N - 1)]
    for p in range(N - 1):
        a[p][msg_id] = pi[p]

    # Step 2: b, the side-information sum vector.
    b = {i: rng.randrange(1, N) for i in S}

    # Step 3: sample (I, J), pick R subset of S and build b_0 / b_1..b_I.
    I, J = sample_query_parameters(M, K, N, rng)
    R = rng.sample(S, I) if I > 0 else []
    excluded = list(R)
    rng.shuffle(excluded)  # excluded[p] is removed from R to form R_p

    # Step 4: theta with P(theta=0) = (M-I)/(M-I+1).
    p_theta0 = (M - I) / (M - I + 1)
    theta = 0 if rng.random() < p_theta0 else 1

    # Step 5: c over a J-subset T of the interference messages.
    interference = [i for i in range(K) if i != msg_id and i not in S]
    T = rng.sample(interference, J) if J > 0 else []
    c = {i: rng.randrange(1, N) for i in T}

    # Step 6: build u_1 .. u_N.
    u = [np.zeros(K, dtype=np.int64) for _ in range(N)]
    for i in R:
        u[0][i] = b[i]
    for i, val in c.items():
        u[0][i] = val

    for p in range(N - 1):
        u[p + 1][msg_id] = pi[p]
        if theta == 1 and p < I:
            support = set(R) - {excluded[p]}
        else:
            support = set(S)
        for i in support:
            u[p + 1][i] = b[i]
        for i, val in c.items():
            u[p + 1][i] = val

    # Step 7: apply a random server permutation sigma.
    sigma = rng.sample(range(N), N)
    vectors = [np.array(u[sigma[n]], copy=True) for n in range(N)]

    return QueryPlan(
        vectors=vectors,
        demand=msg_id,
        side_info_ids=S,
        K=K,
        N=N,
        M=M,
        I=I,
        J=J,
        theta=theta,
        pi=pi,
        sigma=sigma,
        b=b,
        R=R,
        excluded=excluded,
        c=c,
    )


def generate_query_vectors(
    msg_id: int,
    side_info_ids: Sequence[int],
    K: int,
    N: int,
    M: int,
    rng: random.Random,
) -> List[np.ndarray]:
    """Return only the ``N`` query vectors of one PIR-PSI realization.

    Thin wrapper around :func:`build_query` for callers that do not need the
    decoding metadata.

    Args:
        msg_id: The demanded message id.
        side_info_ids: The ``M`` side-information ids.
        K: Total number of messages.
        N: Number of servers.
        M: Number of side-information messages.
        rng: Seeded ``random.Random`` instance.

    Returns:
        A list of ``N`` ``np.ndarray`` of shape ``(K,)`` with entries in
        ``[0, N-1]``.
    """
    return build_query(msg_id, side_info_ids, K, N, M, rng).vectors
