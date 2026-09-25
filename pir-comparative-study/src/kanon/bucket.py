"""Prefix-based bucket construction for k-anonymity query hiding.

Messages are grouped into buckets by the leading digits of their ids (when
the id is written in a fixed base/width).  A client asking for message id
``x`` only sends the *prefix* that ``x`` shares with ``k - 1`` other ids; the
server cannot tell which member of the bucket was actually wanted.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.common.message import Message


def prefix_width(K: int, base: int = 2) -> int:
    """Number of digits needed to write any id ``0 .. K-1`` in ``base``.

    Args:
        K: Number of distinct message ids.
        base: Radix used for the id representation (default 2).

    Returns:
        The digit width ``ceil(log_base(K))`` (``1`` when ``K <= 1``).
    """
    if K <= 1:
        return 1
    return int(math.ceil(math.log(K, base)))


def compute_prefix(msg_id: int, prefix_len: int, base: int, width: int) -> str:
    """Return the first ``prefix_len`` digits of ``msg_id``.

    ``msg_id`` is written in ``base`` using exactly ``width`` digits (with
    leading zeros), then the first ``prefix_len`` of those digits are kept.

    Args:
        msg_id: The id to prefix.
        prefix_len: Number of leading digits to keep.
        base: Radix of the id representation.
        width: Fixed digit width of the id representation.

    Returns:
        The prefix string (length ``min(prefix_len, width)``).
    """
    if prefix_len <= 0:
        raise ValueError(f"prefix_len must be >= 1, got {prefix_len}")
    digits = []
    value = max(0, int(msg_id))
    for _ in range(width):
        value, digit = divmod(value, base)
        digits.append(str(digit))
    digits.reverse()  # most significant first
    return "".join(digits)[:prefix_len]


def build_buckets(
    messages: List[Message], prefix_len: int, base: int = 2
) -> Tuple[Dict[str, List[int]], int]:
    """Group message ids into buckets by prefix.

    Args:
        messages: Dataset to group.
        prefix_len: Number of leading id digits shared by a bucket.
        base: Radix of the id representation.

    Returns:
        A tuple ``(buckets, width)`` where ``buckets`` maps each observed
        prefix to the sorted list of ids that share it, and ``width`` is the
        digit width used to encode ids.
    """
    K = max(len(messages), 1)
    width = prefix_width(K if K else 1, base)
    effective = min(width, max(1, int(prefix_len)))
    buckets: Dict[str, List[int]] = {}
    for msg in messages:
        prefix = compute_prefix(msg.msg_id, effective, base, width)
        buckets.setdefault(prefix, []).append(msg.msg_id)
    for ids in buckets.values():
        ids.sort()
    return buckets, width


def choose_bucket(
    msg_id: int, buckets: Dict[str, List[int]]
) -> Tuple[str, List[int]]:
    """Pick the bucket that contains ``msg_id``.

    Args:
        msg_id: The id whose bucket is wanted.
        buckets: Prefix -> member id list mapping.

    Returns:
        ``(prefix, member_ids)`` where ``prefix`` is the bucket key and
        ``member_ids`` are the ids sharing it.

    Raises:
        ValueError: If no bucket contains ``msg_id``.
    """
    for prefix, member_ids in buckets.items():
        if msg_id in member_ids:
            return prefix, member_ids
    raise ValueError(f"no bucket contains msg_id {msg_id}")