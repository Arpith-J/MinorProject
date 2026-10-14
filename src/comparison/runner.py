"""Runs all four retrieval approaches on the same dataset and collects metrics.

Every approach is reduced to the same four metrics (privacy score, bytes
exchanged, operation counts, latency) so the comparison is uniform and can be
plotted side by side.
"""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.common.dataset import load_csv, load_synthetic
from src.common.message import Message
from src.common.metrics import (
    communication_cost,
    computation_cost,
    identity_bits,
    latency,
    privacy_score,
)
from src.direct.client import DirectClient
from src.direct.server import DirectServer
from src.kanon.client import KAnonClient
from src.kanon.server import KAnonServer
from src.pir.client import PIRClient
from src.pir.server import PIRServer
from src.pir_psi.capacity import capacity_rate
from src.pir_psi.client import PIRPSIClient
from src.pir_psi.server import PIRPSIServer

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> Dict[str, object]:
    """Load the YAML configuration file.

    Args:
        path: Path to a ``config.yaml`` file.

    Returns:
        The configuration as a nested dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def default_config() -> Dict[str, object]:
    """Return a built-in configuration when none is supplied."""
    return {
        "N": 2,
        "K": 8,
        "M": 2,
        "L": 16,
        "q": 2,
        "dataset_path": None,
        "random_seed": 42,
        "latency_repeats": 5,
        "kanon": {"prefix_len": 1, "base": 2},
    }


def prepare_dataset(config: Dict[str, object]) -> List[Message]:
    """Load the dataset for the given configuration.

    If ``config["dataset_path"]`` points to an existing CSV it is loaded,
    otherwise a reproducible synthetic dataset is generated.

    Args:
        config: Configuration dict with at least ``K``, ``L``, ``q``,
            ``dataset_path`` and ``random_seed``.

    Returns:
        The list of messages.
    """
    dataset_path = config.get("dataset_path")
    if dataset_path and Path(dataset_path).is_file():
        return load_csv(dataset_path)
    return load_synthetic(
        K=int(config["K"]),
        L=int(config["L"]),
        q=int(config.get("q", 2)),
        seed=int(config["random_seed"]),
    )


def _resolve_side_info(config: Dict[str, object], msg_id: int) -> List[int]:
    """Choose ``M`` random side-information ids that exclude ``msg_id``."""
    K = int(config["K"])
    M = int(config.get("M", 0))
    rng = random.Random(int(config["random_seed"]))
    pool = [i for i in range(K) if i != msg_id]
    rng.shuffle(pool)
    return pool[: min(M, len(pool))]


def _min_latency(fn, repeats: int) -> float:
    """Return the minimum wall-clock time of ``repeats`` invocations of ``fn``."""
    best = math.inf
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        fn()
        best = min(best, latency(start, time.perf_counter()))
    return best


def _recovered_ok(messages: List[Message], msg_id: int, client_result: Message) -> bool:
    """Return whether a retrieval result is bit-identical to the dataset copy."""
    expected = next(m for m in messages if m.msg_id == msg_id)
    return bool(np.array_equal(expected.payload, client_result.payload))


def _run_direct(messages: List[Message], msg_id: int, config) -> Dict[str, object]:
    repetitions = int(config.get("latency_repeats", 1))
    server = DirectServer(messages)
    client = DirectClient(server)
    message_bytes = len(messages[0].to_bytes())

    def op() -> None:
        client.retrieve(msg_id)

    elapsed = _min_latency(op, repetitions)
    K = len(messages)
    return {
        "privacy": {
            "leakage_bits": identity_bits(K),
            "privacy_score": privacy_score(identity_bits(K), identity_bits(K)),
        },
        "communication": communication_cost(4, message_bytes),
        "computation": [computation_cost(1, "lookup")],
        "latency_s": elapsed,
        "recovered": _recovered_ok(messages, msg_id, client.retrieve(msg_id)),
    }


def _run_kanon(messages: List[Message], msg_id: int, config) -> Dict[str, object]:
    repetitions = int(config.get("latency_repeats", 1))
    kanon_cfg = config.get("kanon", {})
    prefix_len = int(kanon_cfg.get("prefix_len", 1))
    base = int(kanon_cfg.get("base", 2))
    server = KAnonServer(messages, prefix_len=prefix_len, base=base)
    client = KAnonClient(server, known_ids=[m.msg_id for m in messages], prefix_len=prefix_len)
    msg_bytes = len(messages[0].to_bytes())

    def op() -> None:
        client.retrieve(msg_id)

    elapsed = _min_latency(op, repetitions)
    k = server.bucket_size(msg_id)
    prefix = client._prefix_of(msg_id)  # noqa: SLF001 - reused identifier
    K = len(messages)
    return {
        "privacy": {
            "leakage_bits": math.log2(max(1, k)),
            "privacy_score": privacy_score(math.log2(max(1, k)), identity_bits(K)),
        },
        "communication": communication_cost(len(prefix.encode("utf-8")), k * msg_bytes),
        "computation": [computation_cost(k, "bucket_filter")],
        "latency_s": elapsed,
        "recovered": _recovered_ok(messages, msg_id, client.retrieve(msg_id)),
        "bucket_size": k,
    }


def _run_pir(
    messages: List[Message], msg_id: int, side_info_ids: Sequence[int], config
) -> Dict[str, object]:
    repetitions = int(config.get("latency_repeats", 1))
    N = int(config["N"])
    servers = [PIRServer(messages, server_id=i) for i in range(N)]
    client = PIRClient(servers, seed=int(config["random_seed"]))

    def op() -> None:
        client.retrieve(msg_id, list(side_info_ids))

    elapsed = _min_latency(op, repetitions)
    recovered = client.retrieve(msg_id, list(side_info_ids))
    comm = client.last_communication()
    stats = client.last_stats or {}
    PIR_max = comm["total_bytes"] + int(N * int(config["K"]))
    return {
        "privacy": {
            "leakage_bits": 0.0,
            "privacy_score": 1.0,
        },
        "communication": {"bytes_sent": comm["bytes_sent"], "bytes_received": comm["bytes_received"], "total_bytes": comm["total_bytes"]},
        "computation": [
            computation_cost(int(stats.get("xor_ops", 0)), "xors"),
            computation_cost(int(N * int(config["K"])), "query_gen"),
        ],
        "latency_s": elapsed,
        "recovered": _recovered_ok(messages, msg_id, recovered),
        "pir_overhead_max": PIR_max,
    }


def _choose_pir_psi_N(config: Dict[str, object], M: int, L: int) -> int:
    """Pick a server count usable by PIR-PSI (``N >= M+1``, ``(N-1) | L``).

    The configured ``N`` is used when it already satisfies the constraints;
    otherwise the smallest valid ``N`` is chosen.

    Raises:
        ValueError: If no ``N`` in ``[max(2, M+1), 64]`` makes ``L`` divisible
            by ``N-1``.
    """
    configured = int(config.get("N", 2))
    minimum = max(2, M + 1)
    candidates = [n for n in range(minimum, 65) if L % (n - 1) == 0]
    if not candidates:
        raise ValueError(
            f"no PIR-PSI server count N >= {minimum} divides L={L} into N-1 parts"
        )
    if configured in candidates:
        return configured
    return candidates[0]


def _run_pir_psi(
    messages: List[Message],
    msg_id: int,
    side_info_ids: Sequence[int],
    config: Dict[str, object],
) -> Dict[str, object]:
    """Run the PIR-PSI approach and collect its metrics."""
    repetitions = int(config.get("latency_repeats", 1))
    K = len(messages)
    L = messages[0].length
    M = len(side_info_ids)
    N = _choose_pir_psi_N(config, M, L)
    servers = [PIRPSIServer(i, messages, N=N) for i in range(N)]
    side_set = set(side_info_ids)
    side_msgs = [m for m in messages if m.msg_id in side_set]
    client = PIRPSIClient(
        K=K, N=N, M=M, L=L, q=int(config.get("q", 2)),
        side_info_ids=list(side_info_ids), side_info_messages=side_msgs,
        seed=int(config.get("random_seed", 42)),
    )

    def op() -> None:
        client.retrieve(msg_id, servers)

    elapsed = _min_latency(op, repetitions)
    recovered = client.retrieve(msg_id, servers)
    comm = client.last_communication()
    stats = client.last_stats or {}
    downloaded = float(stats.get("downloaded_bits", 0) or 0)
    measured_rate = (L / downloaded) if downloaded > 0 else 0.0
    cap = capacity_rate(N, K, M)
    return {
        "privacy": {"leakage_bits": 0.0, "privacy_score": 1.0},
        "communication": {
            "bytes_sent": comm["bytes_sent"],
            "bytes_received": comm["bytes_received"],
            "total_bytes": comm["total_bytes"],
        },
        "computation": [
            computation_cost(int(stats.get("xor_ops", 0)), "xors"),
            computation_cost(int(N * K), "query_gen"),
        ],
        "latency_s": elapsed,
        "recovered": _recovered_ok(messages, msg_id, recovered),
        "rate": measured_rate,
        "capacity_rate": cap,
        "rate_ratio": (measured_rate / cap) if cap > 0 else 0.0,
        "servers": N,
        "sub_packetization": N - 1,
        "side_info_used": M,
    }


def run_pir_psi(
    msg_id: int,
    side_info_ids: Optional[Sequence[int]] = None,
    config: Optional[Dict[str, object]] = None,
    messages: Optional[List[Message]] = None,
) -> Dict[str, object]:
    """Public entry point for the PIR-PSI approach.

    Args:
        msg_id: The message to retrieve.
        side_info_ids: The private side-information ids (defaults to ``M``
            random ids chosen with the config seed).
        config: Configuration dict (defaults to :func:`default_config`).
        messages: Optional pre-loaded dataset (otherwise loaded from config).

    Returns:
        The PIR-PSI metrics dict (same keys as the other approaches plus
        ``rate``, ``capacity_rate`` and ``servers``).
    """
    cfg = config if config is not None else default_config()
    K = int(cfg["K"])
    if not 0 <= msg_id < K:
        raise ValueError(f"msg_id {msg_id} out of range for K={K}")
    side = list(side_info_ids) if side_info_ids is not None else _resolve_side_info(cfg, msg_id)
    data = messages if messages is not None else prepare_dataset(cfg)
    return _run_pir_psi(data, msg_id, side, cfg)


def run_all(
    msg_id: Optional[int] = None,
    side_info_ids: Optional[Sequence[int]] = None,
    config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run direct access, k-anonymity, two-server PIR and PIR-PSI.

    Args:
        msg_id: The message to retrieve (defaults to ``K // 2``).
        side_info_ids: Side information ids for PIR (defaults to ``M``
            random ids drawn with the config seed).
        config: Configuration dict (defaults to :func:`default_config`).

    Returns:
        A structured result with the configuration, the chosen ids and a
        per-approach metrics dict.
    """
    cfg = config if config is not None else default_config()
    K = int(cfg["K"])
    target = int(msg_id) if msg_id is not None else K // 2
    if not 0 <= target < K:
        raise ValueError(f"msg_id {target} out of range for K={K}")
    side = list(side_info_ids) if side_info_ids is not None else _resolve_side_info(cfg, target)
    messages = prepare_dataset(cfg)

    result: Dict[str, object] = {
        "config": cfg,
        "msg_id": target,
        "side_info_ids": side,
        "approaches": {
            "direct": _run_direct(messages, target, cfg),
            "kanon": _run_kanon(messages, target, cfg),
            "pir": _run_pir(messages, target, side, cfg),
            "pir_psi": _run_pir_psi(messages, target, side, cfg),
        },
    }
    logger.info("comparison run complete for msg_id=%d", target)
    return result


def summarize(result: Dict[str, object]):
    """Flatten a ``run_all`` result into a pandas DataFrame.

    Columns: ``approach``, ``privacy_score``, ``leakage_bits``,
    ``bytes_sent``, ``bytes_received``, ``total_bytes``, ``latency_s``.

    Args:
        result: The dict returned by :func:`run_all`.

    Returns:
        A pandas DataFrame with one row per approach.
    """
    import pandas as pd

    rows = []
    for name, metrics in result["approaches"].items():
        rows.append(
            {
                "approach": name,
                "privacy_score": metrics["privacy"]["privacy_score"],
                "leakage_bits": metrics["privacy"]["leakage_bits"],
                "bytes_sent": metrics["communication"]["bytes_sent"],
                "bytes_received": metrics["communication"]["bytes_received"],
                "total_bytes": metrics["communication"]["total_bytes"],
                "latency_s": metrics["latency_s"],
            }
        )
    return pd.DataFrame(rows)