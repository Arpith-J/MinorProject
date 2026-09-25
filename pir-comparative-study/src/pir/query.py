"""Randomized query-vector generation for the two-server XOR PIR scheme.

The scheme (Chor et al., 1995): to retrieve message ``d`` the client picks a
uniformly random subset ``S`` of the messages it does **not** already have
and sends the indicator vector of ``S`` to server 0 and the indicator vector
of ``S △ {d}`` to server 1.  Individually each vector is perfectly uniform
(independent of ``d``), so neither server can distinguish ``d`` from any
other candidate.

The all-zero vector may arise with probability ``2^-K``; servers then return
no answer (:meth:`~src.pir.server.PIRServer.answer` returns ``None``) and the
client treats a missing answer as the zero block.  Requiring the vectors to
be *always* non-zero would leak information (it would forbid ``e_d`` from
appearing on server 1), so information-theoretic privacy is preserved by
allowing the rare zero case and handling it transparently.
"""

from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np


def _uniform_vector(K: int, rng: random.Random) -> np.ndarray:
    """Draw a uniform ``0/1`` vector of length ``K``.

    Args:
        K: Vector length.
        rng: Seeded ``random.Random`` instance.

    Returns:
        A fresh ``np.ndarray`` of ``np.int8``.
    """
    return np.asarray([rng.randrange(2) for _ in range(K)], dtype=np.int8)


def generate_query_vectors(
    msg_id: int,
    side_info_ids: Sequence[int],
    N: int,
    K: int,
    rng: random.Random,
) -> List[np.ndarray]:
    """Generate the per-server query vectors for one private retrieval.

    Args:
        msg_id: The id of the message the client wants.
        side_info_ids: Ids the client already holds (revealed to the servers
            as *non-private* side information and excluded from both queries).
        N: Number of servers.  Servers 0 and 1 run the active XOR protocol;
            any extra servers receive independent random vectors so their
            views stay independent of ``msg_id``.
        K: Total number of messages in the dataset.
        rng: Seeded ``random.Random`` for reproducibility.

    Returns:
        A list of ``N`` numpy arrays of shape ``(K,)`` with 0/1 entries.
        Each server's vector is uniformly random, so its marginal view is
        independent of ``msg_id``.

    Raises:
        ValueError: On invalid parameters (``msg_id`` out of range, side
            information overlapping the desired id, ``K - |S| < 2``).
    """
    if N < 2:
        raise ValueError(f"need at least 2 servers, got {N}")
    if not 0 <= msg_id < K:
        raise ValueError(f"msg_id {msg_id} out of range for K={K}")
    if K < 2:
        raise ValueError(f"need at least 2 messages, got K={K}")
    side = [int(s) for s in side_info_ids]
    if msg_id in side:
        raise ValueError(f"msg_id {msg_id} appears in side information")
    for s in side:
        if not 0 <= s < K:
            raise ValueError(f"side info id {s} out of range for K={K}")
    active = [m for m in range(K) if m not in side]
    if len(active) < 2:
        raise ValueError(
            f"need at least 2 non-side-info messages, got {len(active)} "
            f"(K={K}, |side_info|={len(side)})"
        )

    e_d = np.zeros(K, dtype=np.int8)
    e_d[msg_id] = 1

    # Active pair: q0 uniform, q1 = q0 XOR e_d.  Both marginal distributions
    # are exactly uniform over {0,1}^K and therefore independent of msg_id.
    q0 = _uniform_vector(K, rng)
    for m in range(K):  # mask out revealed side information
        if m not in active:
            q0[m] = 0
    q1 = np.bitwise_xor(q0, e_d).astype(np.int8)

    vectors = [q0, q1]
    for _ in range(max(0, N - 2)):
        dummy = _uniform_vector(K, rng)
        for m in range(K):
            if m not in active:
                dummy[m] = 0
        vectors.append(dummy)
    return vectors


def is_nonzero(query: np.ndarray) -> bool:
    """Return ``True`` iff at least one entry of ``query`` is non-zero."""
    return bool(np.any(query))