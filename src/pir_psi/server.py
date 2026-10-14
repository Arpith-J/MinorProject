"""PIR-PSI server: answers sub-packet selection queries with XOR sums.

Each server stores identical copies of the whole dataset.  A query is a
vector :math:`\\mathrm{v} \\in [0:N-1]^K`: entry ``v[i]`` selects sub-packet
``v[i]`` of message ``i``, where sub-packet ``0`` is the degenerate all-zero
block.  The answer is the field sum (XOR over :math:`\\mathbb{F}_2`) of the
selected sub-packets, or nothing if the query is all-zero.

Because the query is drawn from the non-uniform distribution of Erhili 2024
(which depends only on the query support size), each server's view is
independent of the demand and of the side-information set.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from src.common.message import Message

logger = logging.getLogger(__name__)


def _field_add(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Add two payloads over ``GF(q)`` (XOR when ``q == 2``)."""
    if q == 2:
        return np.bitwise_xor(a, b).astype(np.uint8)
    return ((a.astype(np.int64) + b.astype(np.int64)) % q).astype(np.uint8)


class PIRPSIServer:
    """One of the ``N`` non-colluding PIR-PSI servers."""

    def __init__(
        self,
        server_id: int,
        messages: List[Message],
        N: int,
        rng: object = None,
    ) -> None:
        """Store the dataset and pre-split every message into ``N-1`` parts.

        Args:
            server_id: Index of this server in ``[0, N-1]``.
            messages: The full dataset (each message's length must be
                divisible by ``N-1``).
            N: Number of servers; the sub-packetization level is ``N-1``.
            rng: Unused placeholder kept for interface symmetry with the
                other servers; accepted and ignored.

        Raises:
            ValueError: If ``N < 2``, the dataset is empty, or a message
                length is not divisible by ``N-1``.
        """
        del rng  # servers are deterministic; accepted for API symmetry
        if N < 2:
            raise ValueError(f"need at least 2 servers, got N={N}")
        if not messages:
            raise ValueError("server needs a non-empty dataset")
        self.server_id = int(server_id)
        self.N = int(N)
        self._messages: Dict[int, Message] = {m.msg_id: m for m in messages}
        self._ordered: List[Message] = [self._messages[i] for i in sorted(self._messages)]
        self._q = self._ordered[0].q
        self._sub_len = self._ordered[0].length // (N - 1)
        self._subs: Dict[int, List[Message]] = {}
        for msg in self._ordered:
            sub = msg.split_subpackets(N - 1)
            self._subs[msg.msg_id] = sub
        self.log: List[Dict[str, object]] = []

    def answer(self, query_vector: np.ndarray) -> Optional[np.ndarray]:
        """Return the XOR sum of the selected message sub-packets.

        Args:
            query_vector: Integer vector of shape ``(K,)`` with entries in
                ``[0, N-1]``.

        Returns:
            The field sum of the selected sub-packets (shape
            ``(L/(N-1),)``), or ``None`` if the query is all-zero.

        Raises:
            ValueError: If the query length or an entry is out of range.
        """
        query = np.asarray(query_vector)
        if query.shape[0] != len(self._ordered):
            raise ValueError(
                f"query length {query.shape[0]} != dataset size {len(self._ordered)}"
            )
        selected: List[np.ndarray] = []
        for msg, val in zip(self._ordered, query):
            vi = int(val)
            if vi < 0 or vi > self.N - 1:
                raise ValueError(f"query entry {vi} outside [0, {self.N - 1}]")
            if vi == 0:
                continue
            selected.append(self._subs[msg.msg_id][vi - 1].payload)
        self.log.append(
            {"server_id": self.server_id, "query": query.astype(int).tolist()}
        )
        if not selected:
            logger.info("pir-psi server %d: all-zero query -> no answer", self.server_id)
            return None
        out = selected[0].copy()
        for other in selected[1:]:
            out = _field_add(out, other, self._q)
        logger.info(
            "pir-psi server %d: answered sum of %d sub-packets",
            self.server_id,
            len(selected),
        )
        return out.astype(np.uint8)

    def answer_size_bytes(self) -> int:
        """Size in bytes of a single non-empty answer sub-packet."""
        return int(self._sub_len)

    def answer_size_bits(self) -> int:
        """Information content of a single non-empty answer, in bits."""
        import math

        return int(self._sub_len * math.log2(self._q)) if self._q > 1 else 0
