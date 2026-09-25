"""Two-server PIR client: generates query vectors, decodes the answers.

The client holds a subset of the dataset as *side information* and uses two
non-colluding servers.  Neither server learns which message is wanted; the
client XORs the two answers to recover the requested message bit-exactly.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.common.message import Message
from src.pir.query import generate_query_vectors
from src.pir.server import PIRServer


class PIRClient:
    """Client that privately retrieves one message from two PIR servers."""

    def __init__(self, servers: Sequence[PIRServer], seed: int = 42) -> None:
        """Bind the client to the servers.

        Args:
            servers: The PIR servers; at least 2 are required, servers 0 and 1
                run the active protocol.
            seed: Seed for the client's ``random.Random`` instance.

        Raises:
            ValueError: If fewer than two servers are supplied.
        """
        servers = list(servers)
        if len(servers) < 2:
            raise ValueError(f"need at least 2 PIR servers, got {len(servers)}")
        self.servers = servers
        self.rng = random.Random(seed)
        self.last_stats: Optional[Dict[str, object]] = None

    def _new_rng(self) -> random.Random:
        return random.Random(self.rng.getrandbits(32))

    def retrieve(self, msg_id: int, side_info_ids: Sequence[int]) -> Message:
        """Privately retrieve the message with the given id.

        Args:
            msg_id: The id of the message to fetch.
            side_info_ids: Ids the client already holds (shared with the
                servers as non-private side information).

        Returns:
            The exact :class:`Message` whose payload is bit-identical to the
            dataset copy.

        Raises:
            RuntimeError: If any server fails to answer (query degeneracy).
        """
        start = time.perf_counter()
        K = len(self.servers[0]._ordered)  # dataset size on server 0
        L = self.servers[0]._ordered[0].length
        self._length = L
        queries = generate_query_vectors(
            msg_id, list(side_info_ids), len(self.servers), K, self._new_rng()
        )
        sent_bytes = sum(int(q.nbytes) for q in queries)

        answers: List[Optional[np.ndarray]] = []
        for server, query in zip(self.servers, queries):
            answers.append(server.answer(query))

        received_bytes = sum(0 if a is None else int(a.nbytes) for a in answers)
        ans0 = answers[0] if answers[0] is not None else np.zeros(self._length, dtype=np.uint8)
        ans1 = answers[1] if answers[1] is not None else np.zeros(self._length, dtype=np.uint8)
        recovered = np.bitwise_xor(ans0, ans1).astype(np.uint8)

        xor_ops = int(np.count_nonzero(queries[0])) + int(np.count_nonzero(queries[1]))
        elapsed = time.perf_counter() - start

        result = Message(msg_id, recovered, q=2)
        self.last_stats = {
            "bytes_sent": sent_bytes,
            "bytes_received": received_bytes,
            "xor_ops": xor_ops,
            "query_vectors": [q.astype(int).tolist() for q in queries],
            "latency_s": elapsed,
        }
        return result

    def privacy_leakage_bits(self) -> float:
        """Number of identity bits leaked to any single PIR server (0.0)."""
        return 0.0

    def last_communication(self) -> Dict[str, float]:
        """Return the bytes exchanged during the most recent retrieval.

        Raises:
            RuntimeError: If no retrieval has happened yet.
        """
        if self.last_stats is None:
            raise RuntimeError("no retrieval performed yet")
        return {
            "bytes_sent": int(self.last_stats["bytes_sent"]),
            "bytes_received": int(self.last_stats["bytes_received"]),
            "total_bytes": int(self.last_stats["bytes_sent"])
            + int(self.last_stats["bytes_received"]),
        }