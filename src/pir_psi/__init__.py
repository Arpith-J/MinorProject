"""PIR with Private Side Information (Erhili & Heidarzadeh, 2024).

This package implements the capacity-achieving PIR-PSI scheme with
sub-packetization :math:`N-1` that uses only XOR operations (no MDS codes).
It is a sibling of :mod:`src.pir`, which implements the classical Chor
two-server XOR PIR and remains the :math:`M = 0` baseline.

Sub-modules:
    query: Non-uniform randomized query generation (Erhili 2024, Alg. 1).
    server: Sub-packet answering server.
    client: Query generation plus retrieval orchestration.
    decoder: Side-information cancellation and demand recovery.
    capacity: Rate formula and achievability conditions.
"""

from __future__ import annotations

from src.pir_psi.capacity import (
    capacity_rate,
    expected_download_bits,
    is_achievable,
    measure_actual_rate,
)
from src.pir_psi.client import PIRPSIClient
from src.pir_psi.decoder import decode
from src.pir_psi.query import (
    QueryPlan,
    build_query,
    combinatorial_m,
    generate_query_vectors,
    permute_vectors,
    probability_matrix,
    sample_query_parameters,
)
from src.pir_psi.server import PIRPSIServer

__all__ = [
    "QueryPlan",
    "build_query",
    "combinatorial_m",
    "generate_query_vectors",
    "permute_vectors",
    "probability_matrix",
    "sample_query_parameters",
    "PIRPSIServer",
    "PIRPSIClient",
    "decode",
    "capacity_rate",
    "is_achievable",
    "expected_download_bits",
    "measure_actual_rate",
]
