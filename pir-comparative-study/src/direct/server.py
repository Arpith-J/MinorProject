"""Direct-access server: returns the exact message for a given id.

This is the privacy baseline.  The server must know the full ``msg_id`` and
logs every query, demonstrating that the client has **zero privacy**.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from src.common.message import Message

logger = logging.getLogger(__name__)


class DirectServer:
    """Dataset holder that answers point lookups by message id."""

    def __init__(self, messages: List[Message]) -> None:
        """Store the messages in a lookup dict keyed by id.

        Args:
            messages: The dataset served by this server.
        """
        self._messages: Dict[int, Message] = {m.msg_id: m for m in messages}
        self.log: List[Dict[str, int]] = []

    def query(self, msg_id: int) -> Message:
        """Return the exact message whose id equals ``msg_id``.

        The server logs the full id, i.e. it learns exactly which record the
        client wants (zero privacy).

        Args:
            msg_id: The id of the requested message.

        Returns:
            The requested :class:`Message`.

        Raises:
            KeyError: If ``msg_id`` is not present in the dataset.
        """
        if msg_id not in self._messages:
            raise KeyError(f"unknown message id: {msg_id}")
        self.log.append({"msg_id": int(msg_id)})
        logger.info("direct query: msg_id=%d (privacy fully leaked)", int(msg_id))
        return self._messages[int(msg_id)]

    def __len__(self) -> int:
        """Number of messages stored."""
        return len(self._messages)