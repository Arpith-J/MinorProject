"""Message model shared by every privacy-preserving retrieval approach.

A :class:`Message` is the atomic unit stored by servers and retrieved by
clients.  It carries an integer id and a numpy payload whose elements are
integers in ``[0, q)``.  For the private retrieval schemes this project
implements, ``q == 2`` so the payload is a bit-vector and all algebra is
vectorized XOR.
"""

from __future__ import annotations

from typing import List

import numpy as np


class Message:
    """A single retrievable record.

    Attributes:
        msg_id: Stable integer identifier of the message.
        payload: ``np.ndarray`` of ``np.uint8`` with entries in ``[0, q)``.
        q: Field size of the message alphabet (default 2).
    """

    def __init__(self, msg_id: int, payload: np.ndarray, q: int = 2) -> None:
        """Initialise a message from an id and a payload.

        Args:
            msg_id: Non-negative integer identifier.
            payload: NumPy array of integers in ``[0, q)``; it is copied so
                later mutation of the caller's array cannot corrupt data.
            q: Field size (``>= 2``) of the message alphabet.

        Raises:
            ValueError: If ``q < 2``  or the payload contains a value outside
                ``[0, q)``.
        """
        if q < 2:
            raise ValueError(f"field size q must be >= 2, got {q}")
        arr = np.asarray(payload, dtype=np.uint8)
        if arr.size and (arr < 0).any() or arr.size and (arr >= q).any():
            raise ValueError(f"payload contains values outside [0, {q})")
        self.msg_id = int(msg_id)
        self.payload = arr.copy()
        self.q = int(q)
        self.length = int(arr.size)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper only
        """Short, unambiguous representation of the message."""
        return f"Message(msg_id={self.msg_id}, length={self.length}, q={self.q})"

    def split_subpackets(self, n: int) -> "List[Message]":
        """Split the payload into ``n`` contiguous sub-packets of equal length.

        Args:
            n: Number of sub-packets (must divide the message length).

        Returns:
            A list of ``n`` new :class:`Message` objects, each sharing this
            message's id, holding a contiguous ``length / n`` slice of the
            payload.

        Raises:
            ValueError: If ``n < 1`` or ``length % n != 0``.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if self.length % n != 0:
            raise ValueError(
                f"message length {self.length} is not divisible by n={n}"
            )
        sub_len = self.length // n
        return [
            Message(self.msg_id, self.payload[i * sub_len:(i + 1) * sub_len], self.q)
            for i in range(n)
        ]

    def to_binary(self) -> np.ndarray:
        """Return the payload as a bit-vector for the ``q == 2`` case.

        Returns:
            A fresh ``np.ndarray`` of ``np.uint8`` bits (0/1) equal to the
            payload.

        Raises:
            ValueError: If this message uses a field size other than ``q == 2``.
        """
        if self.q != 2:
            raise ValueError("to_binary() is only defined for q == 2 messages")
        return self.payload.copy()

    def to_bytes(self) -> bytes:
        """Serialize the payload to raw bytes (used to measure bandwidth).

        Returns:
            ``payload.tobytes()``.
        """
        return self.payload.tobytes()

    @classmethod
    def from_bytes(cls, msg_id: int, data: bytes, q: int = 2) -> "Message":
        """Deserialize a payload previously produced by :meth:`to_bytes`.

        Args:
            msg_id: Identifier to attach to the reconstructed message.
            data: Raw payload bytes produced by :meth:`to_bytes`.
            q: Field size used when the bytes were produced.

        Returns:
            A new :class:`Message`.
        """
        return cls(msg_id, np.frombuffer(data, dtype=np.uint8).astype(np.uint8), q)


def message_bytes(msg: Message) -> int:
    """Return the serialized size of a message in bytes."""
    return len(msg.to_bytes())