"""K-anonymity client: hides the query inside a prefix bucket.

The client knows only the dataset *structure* (the set of ids and the prefix
parameters).  To retrieve message ``x`` it:

1. writes ``x`` in ``base`` with a fixed width;
2. keeps the first ``prefix_len`` digits as the query;
3. sends that prefix to the server and receives the whole bucket;
4. filters the bucket locally on ``msg_id``.
"""

from __future__ import annotations

from typing import List, Sequence

from src.common.message import Message
from src.kanon.bucket import compute_prefix, prefix_width
from src.kanon.server import KAnonServer


class KAnonClient:
    """Client that retrieves a message without revealing its exact id."""

    def __init__(
        self,
        server: KAnonServer,
        known_ids: Sequence[int],
        prefix_len: int = 2,
        base: int = 2,
    ) -> None:
        """Bind the client to a server and to a set of known ids.

        Args:
            server: The :class:`KAnonServer` to query.
            known_ids: All ids present in the dataset (used to compute the id
                width and to filter received buckets).
            prefix_len: Number of leading id digits used for bucketing.
            base: Radix of the id representation.
        """
        self.server = server
        self.known_ids = list(known_ids)
        self.prefix_len = int(prefix_len)
        self.base = int(base)
        K = max(1, len(self.known_ids))
        self._width = prefix_width(K, self.base)
        self._last_bucket_size: int | None = None

    def _prefix_of(self, msg_id: int) -> str:
        return compute_prefix(msg_id, self.prefix_len, self.base, self._width)

    def retrieve(self, msg_id: int) -> Message:
        """Retrieve the message with the given id via its prefix bucket.

        Args:
            msg_id: The id to fetch.

        Returns:
            The exact :class:`Message`.

        Raises:
            ValueError: If ``msg_id`` is not among the known ids.
        """
        if msg_id not in self.known_ids:
            raise ValueError(f"msg_id {msg_id} not among known ids")
        prefix = self._prefix_of(msg_id)
        bucket = self.server.query(prefix)
        self._last_bucket_size = len(bucket)
        for msg in bucket:
            if msg.msg_id == msg_id:
                return msg
        raise ValueError(f"message {msg_id} missing from bucket {prefix}")

    def privacy_level(self) -> int:
        """Return ``k``, the size of the anonymizing bucket used last time.

        Raises:
            RuntimeError: If no retrieval has happened yet.
        """
        if self._last_bucket_size is None:
            raise RuntimeError("no retrieval performed yet")
        return self._last_bucket_size