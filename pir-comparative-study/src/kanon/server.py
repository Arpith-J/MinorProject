"""K-anonymity server: answers prefix queries with a whole bucket of messages.

The server groups the dataset into prefix buckets and, for any prefix query,
returns every message whose id carries that prefix.  It only ever learns the
prefix (never the exact id), which yields ``k``-anonymity where
``k = bucket size``.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from src.common.message import Message
from src.kanon.bucket import build_buckets

logger = logging.getLogger(__name__)


class KAnonServer:
    """Dataset holder that responds only to prefix queries."""

    def __init__(
        self, messages: List[Message], prefix_len: int = 2, base: int = 2
    ) -> None:
        """Store messages and pre-build the prefix buckets.

        Args:
            messages: The dataset to serve.
            prefix_len: Number of leading id digits per bucket.
            base: Radix of the id representation.
        """
        self._messages: Dict[int, Message] = {m.msg_id: m for m in messages}
        self.prefix_len = int(prefix_len)
        self.base = int(base)
        self._buckets, self.width = build_buckets(messages, prefix_len, base)
        self.log: List[Dict[str, str]] = []

    @property
    def buckets(self) -> Dict[str, List[int]]:
        """Prefix -> sorted member ids."""
        return dict(self._buckets)

    def query(self, prefix: str) -> List[Message]:
        """Return every message whose id shares the given prefix.

        Only the prefix is recorded in the server log.

        Args:
            prefix: The bucket prefix to fetch.

        Returns:
            The list of :class:`Message` objects of all ids in that bucket.

        Raises:
            KeyError: If the prefix does not belong to the dataset.
        """
        if prefix not in self._buckets:
            raise KeyError(f"unknown prefix: {prefix}")
        self.log.append({"prefix": prefix})
        logger.info("kanon query: prefix=%s (k=%d)", prefix, len(self._buckets[prefix]))
        return [self._messages[i] for i in self._buckets[prefix]]

    def bucket_size(self, msg_id: int) -> int:
        """Size of the anonymizing bucket containing ``msg_id``."""
        for member_ids in self._buckets.values():
            if msg_id in member_ids:
                return len(member_ids)
        raise ValueError(f"no bucket contains msg_id {msg_id}")