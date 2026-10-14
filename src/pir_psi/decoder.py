"""Side-information cancellation and demand recovery for PIR-PSI.

After the servers answer, the client holds :math:`Y_n` for every server.
Reordering by the inverse of the server permutation :math:`\\sigma` yields
:math:`Z_j`, the answer for vector :math:`u_j`:

* :math:`u_1 = b_0 + c` carries **no** demand sub-packet;
* :math:`u_{p+1} = a_p + (b_p \\text{ or } b) + c` carries demand sub-packet
  :math:`\\pi(p)`.

Because the side-information contributions :math:`b_0`, :math:`b_p` and
:math:`b` are known to the client (it computes them itself), they can be
cancelled exactly:

.. math::
    X_{W,\\pi(p)} = Z_{p+1} - Z_1 - \\text{side}_p + \\text{side}_0,

where :math:`\\text{side}_0` is the sum over :math:`R` (the used
side-information subset) and :math:`\\text{side}_p` is the sum over the
support of the side term in :math:`u_{p+1}`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.pir_psi.query import QueryPlan


def _add(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    if q == 2:
        return np.bitwise_xor(a, b).astype(np.uint8)
    return ((a.astype(np.int64) + b.astype(np.int64)) % q).astype(np.uint8)


def _sub(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    if q == 2:
        return np.bitwise_xor(a, b).astype(np.uint8)
    return ((a.astype(np.int64) - b.astype(np.int64)) % q).astype(np.uint8)


def _side_term(
    ids,
    plan: QueryPlan,
    side_subpackets: Dict[int, List[np.ndarray]],
    sub_len: int,
    q: int,
) -> np.ndarray:
    """Sum the known side-information sub-packets selected by ``ids``."""
    out = np.zeros(sub_len, dtype=np.uint8)
    for i in ids:
        val = plan.b[i]
        out = _add(out, side_subpackets[i][val - 1], q)
    return out


def decode(
    plan: QueryPlan,
    answers: List[Optional[np.ndarray]],
    side_subpackets: Dict[int, List[np.ndarray]],
    L: int,
    q: int = 2,
) -> np.ndarray:
    """Recover the demanded message payload from the server answers.

    Args:
        plan: The :class:`~src.pir_psi.query.QueryPlan` used for the query.
        answers: The ``N`` server answers (``None`` for an all-zero query).
        side_subpackets: Mapping ``id -> [sub-packet 1, ..., sub-packet N-1]``
            for each side-information message (each a ``np.ndarray``).
        L: Length of the demand message in field elements.
        q: Field size (``2`` selects XOR algebra).

    Returns:
        A fresh ``np.ndarray`` of ``np.uint8`` equal to the demand payload.

    Raises:
        ValueError: If the number of answers does not match ``N``.
    """
    N = plan.N
    if len(answers) != N:
        raise ValueError(f"expected {N} answers, got {len(answers)}")
    sub_len = L // (N - 1)

    # Reorder: Z[sigma[n]] = answers[n]  =>  Z[j] = answers[inv[j]].
    inv = [0] * N
    for n, s in enumerate(plan.sigma):
        inv[s] = n
    zero = np.zeros(sub_len, dtype=np.uint8)
    Z = [answers[inv[j]] for j in range(N)]
    Z = [z if z is not None else zero.copy() for z in Z]

    side0 = _side_term(plan.R, plan, side_subpackets, sub_len, q)

    result: List[Optional[np.ndarray]] = [None] * (N - 1)
    for p in range(N - 1):
        if plan.theta == 1 and p < plan.I:
            ids = set(plan.R) - {plan.excluded[p]}
        else:
            ids = set(plan.side_info_ids)
        sidep = _side_term(ids, plan, side_subpackets, sub_len, q)
        x = _sub(Z[p + 1], Z[0], q)
        x = _sub(x, sidep, q)
        x = _add(x, side0, q)
        result[plan.pi[p] - 1] = x

    payload = np.concatenate([r for r in result])  # type: ignore[arg-type]
    return payload.astype(np.uint8)
