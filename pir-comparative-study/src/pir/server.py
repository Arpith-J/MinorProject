"""Two-server PIR server: answers XOR-sum queries over its dataset.

Each server stores the whole dataset and receives a 0/1 query vector.  Its
answer is the bitwise XOR of every message whose query entry is set.  Since
each server receives a uniform random vector (Phase 4 query generation), its
answer carries no information about which message the client actually wants.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from src.common.message import Message

logger = logging.getLogger(__name__)


class PIRServer:
    """One of the (non-colluding) private-information-retrieval servers."""

    def __init__(self, messages: List[Message], server_id: int = 0) -> None:
        """Store the dataset and a server id for book-keeping.

        Args:
            messages: The full dataset served by this server.
            server_id: ``0`` or ``1`` (and up to ``N-1`` for extra servers).
        """
        self.server_id = int(server_id)
        self._messages: Dict[int, Message] = {m.msg_id: m for m in messages}
        self._ordered = [self._messages[i] for i in sorted(self._messages)]
        self.log: List[Dict[str, object]] = []

    def answer_zero(self) -> Optional[np.ndarray]:
        """Return ``None`` for an all-zero query (defensive helper).

        In normal operation :meth:`answer` is only called with non-zero
        vectors (guaranteed by the query generator), so this method simply
        documents and implements the all-zero convention.
        """
        return None

    def answer(self, query: np.ndarray) -> Optional[np.ndarray]:
        """Answer a query with the XOR-sum of the selected messages.

        Args:
            query: A 0/1 vector of shape ``(K,)``; entry ``m`` selects
                message ``m``.

        Returns:
            The bitwise XOR of all selected message payloads (shape ``(L,)``),
            or ``None`` for an all-zero query.

        Raises:
            ValueError: If the query length does not match the dataset size.
        """
        if query.shape[0] != len(self._ordered):
            raise ValueError(
                f"query length {query.shape[0]} != dataset size {len(self._ordered)}"
            )
        selected = [msg.payload for msg, flag in zip(self._ordered, query) if flag]
        self.log.append({"server_id": self.server_id, "query": query.astype(int).tolist()})
        if not selected:
            logger.info("server %d: all-zero query -> no answer", self.server_id)
            return None
        out = selected[0]
        for other in selected[1:]:
            out = np.bitwise_xor(out, other)
        logger.info("server %d: answered XOR of %d messages", self.server_id, len(selected))
        return out.astype(np.uint8)