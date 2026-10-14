"""PIR-PSI client: query generation, retrieval and decoding.

The client knows :math:`M` messages as private side information and wants to
retrieve one more message from :math:`N \\ge M+1` non-colluding servers.  It
generates the randomized queries of Erhili 2024 (Section IV), collects the
answers and cancels the side-information contributions to recover the demand
bit-exactly.  Neither the demand index nor the side-information set is
revealed to any single server.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.common.message import Message
from src.pir_psi.decoder import decode
from src.pir_psi.query import build_query
from src.pir_psi.server import PIRPSIServer

logger = logging.getLogger(__name__)


class PIRPSIClient:
    """Client for the capacity-achieving PIR-PSI scheme."""

    def __init__(
        self,
        K: int,
        N: int,
        M: int,
        L: int,
        q: int = 2,
        side_info_ids: Optional[Sequence[int]] = None,
        side_info_messages: Optional[Sequence[Message]] = None,
        rng: Optional[random.Random] = None,
        seed: int = 42,
        servers: Optional[Sequence[PIRPSIServer]] = None,
    ) -> None:
        """Configure the client.

        Args:
            K: Total number of messages in the dataset.
            N: Number of servers (must satisfy ``N >= M + 1``).
            M: Number of side-information messages the client already holds.
            L: Length of the demand message in field elements.
            q: Field size (``2`` for XOR).
            side_info_ids: The ``M`` ids the client holds.
            side_info_messages: The actual side-information
                :class:`~src.common.message.Message` objects (required when
                ``M > 0``).
            rng: Seeded ``random.Random`` instance.
            seed: Seed used when ``rng`` is not supplied.
            servers: Optional servers to bind at construction time.

        Raises:
            ValueError: If parameters are inconsistent or side-information
                messages are missing while ``M > 0``.
        """
        if N < M + 1:
            raise ValueError(f"PIR-PSI requires N >= M+1, got N={N}, M={M}")
        if K <= M:
            raise ValueError(f"need K > M, got K={K}, M={M}")
        self.K = int(K)
        self.N = int(N)
        self.M = int(M)
        self.L = int(L)
        self.q = int(q)
        self.side_info_ids = [int(s) for s in (side_info_ids or [])]
        if len(self.side_info_ids) != self.M:
            raise ValueError(
                f"expected {self.M} side-info ids, got {len(self.side_info_ids)}"
            )
        self.rng = rng if rng is not None else random.Random(seed)
        self.servers: List[PIRPSIServer] = list(servers) if servers else []
        self.last_stats: Optional[Dict[str, object]] = None

        # Pre-split the side-information messages into N-1 sub-packets.
        self._side_subs: Dict[int, List[np.ndarray]] = {}
        if self.M > 0:
            if not side_info_messages:
                raise ValueError("side_info_messages are required when M > 0")
            by_id = {m.msg_id: m for m in side_info_messages}
            for sid in self.side_info_ids:
                if sid not in by_id:
                    raise ValueError(f"missing side-info message {sid}")
                subs = by_id[sid].split_subpackets(self.N - 1)
                self._side_subs[sid] = [s.payload for s in subs]

    def _new_rng(self) -> random.Random:
        """Derive a fresh reproducible RNG from the master seed."""
        return random.Random(self.rng.getrandbits(32))

    def retrieve(
        self,
        msg_id: int,
        servers: Optional[Sequence[PIRPSIServer]] = None,
    ) -> Message:
        """Privately retrieve the message with the given id.

        Args:
            msg_id: The id of the message to fetch.
            servers: The ``N`` servers to query; defaults to the servers
                bound at construction time.

        Returns:
            The exact :class:`~src.common.message.Message` whose payload is
            identical to the dataset copy.

        Raises:
            ValueError: If the number of servers is not ``N``.
        """
        start = time.perf_counter()
        srv = list(servers) if servers is not None else self.servers
        if len(srv) != self.N:
            raise ValueError(f"expected {self.N} servers, got {len(srv)}")

        plan = build_query(
            msg_id, self.side_info_ids, self.K, self.N, self.M, self._new_rng()
        )
        answers = [server.answer(vec) for server, vec in zip(srv, plan.vectors)]

        payload = decode(plan, answers, self._side_subs, self.L, self.q)

        bytes_sent = sum(int(v.nbytes) for v in plan.vectors)
        bytes_received = sum(0 if a is None else int(a.nbytes) for a in answers)
        downloaded_bits = sum(
            0 if a is None else int(a.size * math.log2(self.q)) for a in answers
        )
        xor_ops = sum(int(np.count_nonzero(v)) for v in plan.vectors)
        elapsed = time.perf_counter() - start

        result = Message(msg_id, payload, q=self.q)
        self.last_stats = {
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "downloaded_bits": downloaded_bits,
            "answers": len([a for a in answers if a is not None]),
            "xor_ops": xor_ops,
            "query_vectors": [v.astype(int).tolist() for v in plan.vectors],
            "I": plan.I,
            "J": plan.J,
            "theta": plan.theta,
            "support_sizes": [plan.support_size(n) for n in range(self.N)],
            "latency_s": elapsed,
        }
        logger.info(
            "pir-psi client: retrieved %d (I=%d, J=%d, theta=%d, answers=%d)",
            msg_id,
            plan.I,
            plan.J,
            plan.theta,
            self.last_stats["answers"],
        )
        return result

    def privacy_leakage_bits(self) -> float:
        """Identity bits leaked to any single server (always ``0.0``)."""
        return 0.0

    def last_communication(self) -> Dict[str, float]:
        """Return the byte counts of the most recent retrieval.

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

    def last_downloaded_bits(self) -> float:
        """Information actually downloaded in the most recent retrieval."""
        if self.last_stats is None:
            raise RuntimeError("no retrieval performed yet")
        return float(self.last_stats["downloaded_bits"])
