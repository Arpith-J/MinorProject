"""Standalone demo of the PIR-PSI scheme (Erhili & Heidarzadeh, 2024).

Uses the paper's illustrative parameters N=4, K=5, M=2, L=12 and prints the
non-uniform probability matrix, one sampled query realization, the per-server
answers and the decoded demand.  ``src/`` uses the ``logging`` module; this
``scripts/`` file is allowed to use ``print``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.dataset import load_synthetic  # noqa: E402
from src.pir_psi.client import PIRPSIClient  # noqa: E402
from src.pir_psi.query import build_query, probability_matrix  # noqa: E402
from src.pir_psi.server import PIRPSIServer  # noqa: E402
from src.pir_psi.capacity import (  # noqa: E402
    capacity_rate,
    expected_download_bits,
    measure_actual_rate,
)


def main() -> None:
    """Run and print a complete PIR-PSI retrieval example."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    N, K, M, L = 4, 5, 2, 12
    msg_id = 2
    side_info_ids = [0, 1]

    print()
    print("=" * 72)
    print("  PIR-PSI DEMO  (Erhili & Heidarzadeh, 2024)")
    print(f"  N={N} servers, K={K} messages, M={M} side-info, L={L} symbols")
    print("=" * 72)

    dataset = load_synthetic(K=K, L=L, q=2, seed=7)
    side_msgs = [dataset[i] for i in side_info_ids]

    print("\n[1] Non-uniform query distribution P[i][j] (Lemma 1):")
    P = probability_matrix(M, K, N)
    header = "      " + "".join(f"  j={j:<6}" for j in range(K - M))
    print(header)
    for i in range(M + 1):
        row = "".join(f"  {P[i, j]:<7.4f}" for j in range(K - M))
        print(f"  i={i} {row}")
    print(f"  sum(P) = {P.sum():.6f}")

    servers = [PIRPSIServer(i, dataset, N=N) for i in range(N)]
    client = PIRPSIClient(
        K=K, N=N, M=M, L=L, q=2,
        side_info_ids=side_info_ids, side_info_messages=side_msgs, seed=42,
    )

    print(f"\n[2] Client holds side information S = {side_info_ids} privately;"
          f" demand W = {msg_id}.")
    plan = build_query(msg_id, side_info_ids, K, N, M, client.rng)
    print(f"    Sampled (I, J) = ({plan.I}, {plan.J}), theta = {plan.theta};"
          f" sub-packetization = N-1 = {N - 1}.")
    print(f"    Demand sub-packet permutation pi = {plan.pi};"
          f" server permutation sigma = {plan.sigma}.")

    print("\n[3] Queries v_n and answers Y_n (sub-packets of length L/(N-1)):")
    for n, v in enumerate(plan.vectors):
        ans = servers[n].answer(v)
        if ans is None:
            print(f"    server {n}: v = {v.tolist()}  ->  no answer (all-zero)")
        else:
            print(f"    server {n}: v = {v.tolist()}  ->  Y_{n} = {ans.tolist()}")

    recovered = client.retrieve(msg_id, servers)
    expected = dataset[msg_id].payload
    ok = np.array_equal(recovered.payload, expected)
    print(f"\n[4] Decoded demand #{recovered.msg_id}:")
    print(f"    payload   = {recovered.payload.tolist()}")
    print(f"    expected  = {expected.tolist()}")
    print(f"    bit-exact = {ok}")

    cap = capacity_rate(N, K, M)
    print("\n[5] Rate:")
    print(f"    capacity rate        R      = {cap:.6f}  (= 16/21)")
    print(f"    expected download    L/R    = {expected_download_bits(N, K, M, L):.4f}")
    print(f"    realized rate (1 run)       = {measure_actual_rate(client, msg_id, servers):.6f}")

    print("\n[6] Privacy: I(W, S; Q_n) = 0 for every server n"
          " (information-theoretic).")
    print()


if __name__ == "__main__":
    main()
