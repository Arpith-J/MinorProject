"""Small step-by-step demo: K=8, N=2, M=2, L=16.

Prints, for each privacy-preserving approach, exactly what the client sends,
what the server answers, and which metric (privacy / communication /
latency) results.  This file is allowed to use ``print`` (scripts), whereas
the packages under ``src/`` use the ``logging`` module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.dataset import load_synthetic  # noqa: E402
from src.comparison.runner import run_all, default_config  # noqa: E402
from src.direct.client import DirectClient  # noqa: E402
from src.direct.server import DirectServer  # noqa: E402
from src.kanon.client import KAnonClient  # noqa: E402
from src.kanon.server import KAnonServer  # noqa: E402
from src.pir.client import PIRClient  # noqa: E402
from src.pir.server import PIRServer  # noqa: E402


def demo_direct(messages, msg_id: int) -> None:
    print("=" * 70)
    print("  APPROACH 1: DIRECT ACCESS (baseline, ZERO privacy)")
    print("=" * 70)
    server = DirectServer(messages)
    client = DirectClient(server)
    print(f"  1. Client asks server for msg [{msg_id}] plainly.")
    msg = client.retrieve(msg_id)
    print(f"  2. Server returns message #{msg.msg_id} (payload = {msg.payload.tolist()}).")
    print(f"  3. Server log: {server.log}  -> the full id is revealed.")
    print("  Privacy: the server knows exactly WHICH message was fetched (0% score).")


def demo_kanon(messages, msg_id: int) -> None:
    print("=" * 70)
    print("  APPROACH 2: PREFIX-BASED K-ANONYMITY")
    print("=" * 70)
    cluster_sizes = [m.msg_id for m in messages]
    server = KAnonServer(messages, prefix_len=1, base=2)
    client = KAnonClient(server, known_ids=cluster_sizes, prefix_len=1)
    prefix = client._prefix_of(msg_id)  # noqa: SLF001 - demo reuse
    print(f"  1. Client computes prefix of msg [{msg_id}]  -> prefix = '{prefix}'")
    print(f"  2. Client sends ONLY the prefix '{prefix}' (the exact id 5 stays local).")
    bucket = server.query(prefix)
    print(f"  3. Server returns {len(bucket)} candidate messages (k={len(bucket)}-anonymity).")
    print(f"  4. First candidate id = {bucket[0].msg_id}; client filters locally for {msg_id}.")
    recovered = client.retrieve(msg_id)
    print(f"  5. Recovered message #{recovered.msg_id} (payload = {recovered.payload.tolist()}).")
    print(f"  Server log: {server.log}  -> never sees {msg_id}.")


def demo_pir(messages, msg_id: int, side_info_ids) -> None:
    print("=" * 70)
    print("  APPROACH 3: TWO-SERVER PIR (XOR scheme)")
    print("=" * 70)
    servers = [PIRServer(messages, 0), PIRServer(messages, 1)]
    client = PIRClient(servers, seed=42)
    print(f"  1. Side information held by the client: {side_info_ids}")
    recovered = client.retrieve(msg_id, side_info_ids)
    stats = client.last_stats
    for s, q in zip(servers, stats["query_vectors"]):
        print(f"  2. Query to server {s.server_id}: {q} (uniform, d-independent)")
    print(f"  3. Server 0 answers   a0; server 1 answers a1; client computes a0 XOR a1.")
    print(f"  4. Recovered message #{recovered.msg_id} (payload = {recovered.payload.tolist()}).")
    print(f"  5. Server logs: {len(servers[0].log)} and {len(servers[1].log)} queries recorded.")
    print("  Privacy: any single server cannot tell WHICH message was fetched (100% score).")


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    print()
    print("DEMO: Comparative study of privacy-preserving information retrieval")
    print("dataset: synthetic  K=8, L=16, q=2 | parameters: N=2, M=2, prefix_len=1")
    messages = load_synthetic(K=8, L=16, q=2, seed=42)
    msg_id = 5
    side_info_ids = [0, 3]

    print()
    demo_direct(messages, msg_id)
    print()
    demo_kanon(messages, msg_id)
    print()
    demo_pir(messages, msg_id, side_info_ids)
    print()
    print("=" * 70)
    print("  COMPARISON SUMMARY (one runner for all three approaches)")
    print("=" * 70)
    cfg = default_config()
    cfg.update({"K": 8, "N": 2, "M": 2, "L": 16, "q": 2, "latency_repeats": 5})
    result = run_all(msg_id=5, side_info_ids=side_info_ids, config=cfg)
    for name, metrics in result["approaches"].items():
        print(
            f"  {name.ljust(7)} | privacy={metrics['privacy']['privacy_score']:.2f} "
            f"| sent={metrics['communication']['bytes_sent']:.0f}B "
            f"| recv={metrics['communication']['bytes_received']:.0f}B "
            f"| latency={metrics['latency_s'] * 1e6:.1f}us "
            f"| recovered={metrics['recovered']!s}"
        )
    print()


if __name__ == "__main__":
    main()