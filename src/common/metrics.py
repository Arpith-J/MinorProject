"""Privacy, communication, computation and latency metrics.

Every retrieval approach is reduced to the same four numbers so that a fair
comparison can be drawn:

* **Privacy** — a score in ``[0, 1]`` derived from the number of identity
  bits the server can infer about the requested message.
* **Communication** — serialized bytes sent/received.
* **Computation** — number of operations weighted by their type.
* **Latency** — wall-clock processing time of the whole retrieval.
"""

from __future__ import annotations

import math
from typing import Dict


def privacy_score(leakage_bits: float, identity_bits: float) -> float:
    """Convert leaked identity bits into a privacy score in ``[0, 1]``.

    ``1.0`` means zero leakage, ``0.0`` means the identity was disclosed
    completely.  The score is ``1 - min(1, leakage / identity)``.

    Args:
        leakage_bits: Number of bits of the target's identity the server can
            infer (``0`` for PIR, ``log2(k)`` for k-anonymity, ``log2(K)``
            for direct access).
        identity_bits: Total number of bits needed to describe the identity,
            normally ``log2(K)`` (``0`` is treated as a perfect score).

    Returns:
        Float in ``[0, 1]``.
    """
    if identity_bits <= 0:
        return 1.0
    if leakage_bits <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - leakage_bits / identity_bits))


def identity_bits(K: int) -> float:
    """Number of bits required to describe one of ``K`` identities."""
    return math.log2(max(1, K))


def communication_cost(bytes_sent: int, bytes_received: int) -> Dict[str, float]:
    """Summarize the serialized bytes exchanged during a retrieval.

    Args:
        bytes_sent: Bytes sent by the client.
        bytes_received: Bytes received by the client.

    Returns:
        Dict with keys ``bytes_sent``, ``bytes_received`` and ``total_bytes``.
    """
    return {
        "bytes_sent": float(bytes_sent),
        "bytes_received": float(bytes_received),
        "total_bytes": float(bytes_sent + bytes_received),
    }


def computation_cost(num_ops: int, op_type: str) -> Dict[str, object]:
    """Summarize the amount of server/client work for an operation type.

    Args:
        num_ops: Number of elementary operations performed.
        op_type: Short label describing the operation, e.g. ``"xors"``,
            ``"bucket_filter"`` or ``"lookup"``.

    Returns:
        Dict with keys ``op_type`` and ``num_ops``.
    """
    return {"op_type": str(op_type), "num_ops": int(num_ops)}


def latency(start_time: float, end_time: float) -> float:
    """Return a wall-clock latency in seconds.

    Args:
        start_time: Timestamp (``time.perf_counter()``) at the start.
        end_time: Timestamp at the end.

    Returns:
        ``end_time - start_time`` clipped below at ``0``.
    """
    return max(0.0, end_time - start_time)