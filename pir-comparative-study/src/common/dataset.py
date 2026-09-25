"""Dataset loading and synthetic generation for the retrieval study.

Every approach in this project operates over the same list of
:class:`~src.common.message.Message` objects.  This module provides a
reproducible synthetic generator and a CSV loader.
"""

from __future__ import annotations

import random
from typing import List, Optional
from pathlib import Path

import numpy as np

from src.common.message import Message


def load_synthetic(K: int, L: int, q: int = 2, seed: int = 42) -> List[Message]:
    """Generate ``K`` random messages of length ``L`` over ``GF(q)``.

    Args:
        K: Number of messages to generate (ids ``0 .. K-1``).
        L: Payload length of every message.
        q: Field size (``2`` produces bit-vectors used by the PIR scheme).
        seed: Random seed for full reproducibility.

    Returns:
        A list of ``K`` :class:`Message` objects.
    """
    rng = random.Random(seed)
    messages: List[Message] = []
    for msg_id in range(K):
        payload = np.asarray([rng.randrange(q) for _ in range(L)], dtype=np.uint8)
        messages.append(Message(msg_id, payload, q=q))
    return messages


def load_csv(path) -> List[Message]:
    """Load messages from a CSV file.

    The file must have a header row ``id,payload`` where ``payload`` is a
    comma-separated list of field elements, e.g.::

        id,payload
        0,0,1,0,1
        1,1,0,1,0

    Args:
        path: Path to the CSV file.

    Returns:
        A list of :class:`Message` objects in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a row cannot be parsed or payload lengths differ.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    messages: List[Message] = []
    lengths: set = set()
    with path.open("r", encoding="utf-8") as fh:
        next(fh)  # skip header
        for line_no, line in enumerate(fh, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            msg_id = int(parts[0])
            payload = np.asarray([int(x) for x in parts[1:]], dtype=np.uint8)
            if payload.size == 0:
                raise ValueError(f"row {line_no}: empty payload")
            lengths.add(payload.size)
            messages.append(Message(msg_id, payload, q=2))
    if len(lengths) > 1:
        raise ValueError(f"payload lengths differ across rows: {sorted(lengths)}")
    return messages


def get_ground_truth(msg_id: int, dataset: Optional[List[Message]] = None) -> Message:
    """Return the message whose id equals ``msg_id``.

    Args:
        msg_id: The id to look up.
        dataset: The dataset to search.  When omitted, a fresh synthetic
            dataset is generated with ``K=16, L=32, q=2, seed=42``.

    Returns:
        The matching :class:`Message`.

    Raises:
        ValueError: If no message with ``msg_id`` exists.
    """
    if dataset is None:
        dataset = load_synthetic(K=16, L=32, q=2, seed=42)
    for msg in dataset:
        if msg.msg_id == msg_id:
            return msg
    raise ValueError(f"ground truth: no message with id {msg_id} in dataset")