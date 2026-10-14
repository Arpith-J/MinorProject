"""Capacity, achievability and rate measurement for PIR-PSI.

The capacity of PIR-PSI with ``N`` servers, ``K`` messages and ``M`` private
side-information messages is (Erhili & Heidarzadeh, 2024, Theorem 1)

.. math::
    R = \\frac{N^{K-M} - N^{K-M-1}}{N^{K-M} - 1},

which is achievable whenever :math:`N \\ge M+1` and :math:`N-1 \\mid L`.
For :math:`M = 0` this is the classical (finite-length) PIR capacity whose
:math:`K \\to \\infty` limit is the Chor rate :math:`(N-1)/N`.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

from src.pir_psi.client import PIRPSIClient
from src.pir_psi.server import PIRPSIServer


def capacity_rate(N: int, K: int, M: int) -> float:
    """Return the PIR-PSI capacity (download rate) :math:`R`.

    Args:
        N: Number of servers.
        K: Total number of messages.
        M: Number of side-information messages.

    Returns:
        The exact rational capacity as a ``float``.

    Raises:
        ValueError: If ``N < M + 1`` or ``K <= M``.
    """
    if N < M + 1:
        raise ValueError(f"need N >= M+1, got N={N}, M={M}")
    if K <= M:
        raise ValueError(f"need K > M, got K={K}, M={M}")
    e = K - M
    numerator = N ** e - N ** (e - 1)
    denominator = N ** e - 1
    return float(Fraction(numerator, denominator))


def is_achievable(N: int, K: int, M: int, L: int) -> bool:
    """Return whether the parameters satisfy the scheme's conditions.

    Achievability requires ``N >= M + 1``, ``K > M`` and ``(N-1) | L`` (so
    that every message splits into ``N-1`` equal sub-packets).

    Args:
        N: Number of servers.
        K: Total number of messages.
        M: Number of side-information messages.
        L: Message length in field elements.

    Returns:
        ``True`` iff the scheme is applicable.
    """
    if N < 2 or K <= M or N < M + 1 or L <= 0:
        return False
    return L % (N - 1) == 0


def expected_download_bits(N: int, K: int, M: int, L: int) -> float:
    """Expected number of downloaded bits :math:`L / R`.

    Args:
        N: Number of servers.
        K: Total number of messages.
        M: Number of side-information messages.
        L: Message length in field elements (bits when ``q == 2``).

    Returns:
        The expected total download, in bits per retrieval.
    """
    return L / capacity_rate(N, K, M)


def measure_actual_rate(
    client: PIRPSIClient,
    msg_id: int,
    servers: Sequence[PIRPSIServer],
) -> float:
    """Run one retrieval and return the realized download rate ``L`` / bits.

    The rate counts only non-empty answers (an all-zero query downloads
    nothing from that server), exactly as in the information-theoretic
    definition of the PIR-PSI rate.

    Args:
        client: The client performing the retrieval.
        msg_id: The message id to retrieve.
        servers: The ``N`` servers to query.

    Returns:
        The realized rate ``L / downloaded_bits`` for this retrieval, or
        ``math.inf`` if nothing was downloaded.
    """
    client.retrieve(msg_id, servers)
    downloaded = client.last_downloaded_bits()
    if downloaded <= 0:
        return math.inf
    return float(client.L) / downloaded
