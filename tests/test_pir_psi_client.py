"""Tests for the PIR-PSI client and decoder (Sub-phase 8c)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.common.dataset import load_synthetic
from src.pir_psi.client import PIRPSIClient
from src.pir_psi.server import PIRPSIServer


def _make_client(dataset, msg_id, M, N, L, seed=42):
    side_ids = [i for i in range(len(dataset)) if i != msg_id][:M]
    side_msgs = [dataset[i] for i in side_ids]
    servers = [PIRPSIServer(i, dataset, N=N) for i in range(N)]
    client = PIRPSIClient(
        K=len(dataset), N=N, M=M, L=L, q=2,
        side_info_ids=side_ids, side_info_messages=side_msgs, seed=seed,
    )
    return client, servers


class TestErhiliExample:
    """The worked example of the paper: N=4, K=5, M=2, L=12."""

    def test_bit_exact_recovery(self):
        N, K, M, L = 4, 5, 2, 12
        dataset = load_synthetic(K=K, L=L, seed=7)
        client, servers = _make_client(dataset, msg_id=2, M=M, N=N, L=L)
        recovered = client.retrieve(2, servers)
        assert recovered.msg_id == 2
        assert np.array_equal(recovered.payload, dataset[2].payload)

    def test_100_random_trials(self):
        N, K, M, L = 4, 5, 2, 12
        dataset = load_synthetic(K=K, L=L, seed=11)
        rng = random.Random(0)
        for trial in range(100):
            msg_id = rng.randrange(K)
            side = rng.sample([i for i in range(K) if i != msg_id], M)
            servers = [PIRPSIServer(i, dataset, N=N) for i in range(N)]
            client = PIRPSIClient(
                K=K, N=N, M=M, L=L, q=2,
                side_info_ids=side, side_info_messages=[dataset[i] for i in side],
                seed=trial,
            )
            recovered = client.retrieve(msg_id, servers)
            assert np.array_equal(recovered.payload, dataset[msg_id].payload), (
                f"trial {trial}, msg_id {msg_id}"
            )


@pytest.mark.parametrize(
    "N,K,M",
    [
        (2, 4, 0),
        (2, 6, 1),
        (2, 8, 1),
        (3, 4, 0),
        (3, 6, 2),
        (3, 8, 1),
        (4, 5, 2),
        (4, 8, 3),
        (4, 6, 0),
    ],
)
def test_recovery_across_parameters(N, K, M):
    L = (N - 1) * 4
    dataset = load_synthetic(K=K, L=L, seed=N * 100 + K * 10 + M)
    rng = random.Random(123)
    for _ in range(20):
        msg_id = rng.randrange(K)
        side = rng.sample([i for i in range(K) if i != msg_id], M)
        servers = [PIRPSIServer(i, dataset, N=N) for i in range(N)]
        client = PIRPSIClient(
            K=K, N=N, M=M, L=L, q=2,
            side_info_ids=side, side_info_messages=[dataset[i] for i in side],
            seed=rng.getrandbits(16),
        )
        recovered = client.retrieve(msg_id, servers)
        assert np.array_equal(recovered.payload, dataset[msg_id].payload)


def test_m_zero_recovery_and_privacy():
    N, K, M, L = 3, 6, 0, 8
    dataset = load_synthetic(K=K, L=L, seed=3)
    servers = [PIRPSIServer(i, dataset, N=N) for i in range(N)]
    client = PIRPSIClient(K=K, N=N, M=0, L=L, q=2, side_info_ids=[])
    for msg_id in range(K):
        recovered = client.retrieve(msg_id, servers)
        assert np.array_equal(recovered.payload, dataset[msg_id].payload)
    assert client.privacy_leakage_bits() == 0.0


def test_stats_and_communication():
    N, K, M, L = 4, 5, 2, 12
    dataset = load_synthetic(K=K, L=L, seed=1)
    client, servers = _make_client(dataset, msg_id=0, M=M, N=N, L=L)
    client.retrieve(0, servers)
    comm = client.last_communication()
    assert comm["total_bytes"] == comm["bytes_sent"] + comm["bytes_received"]
    assert client.last_downloaded_bits() > 0
    assert client.last_stats["answers"] >= 1


def test_client_reuse_is_consistent():
    N, K, M, L = 4, 5, 2, 12
    dataset = load_synthetic(K=K, L=L, seed=2)
    client, servers = _make_client(dataset, msg_id=4, M=M, N=N, L=L)
    for _ in range(10):
        recovered = client.retrieve(4, servers)
        assert np.array_equal(recovered.payload, dataset[4].payload)


def test_requires_side_info_messages():
    N, K, M, L = 4, 5, 2, 12
    with pytest.raises(ValueError):
        PIRPSIClient(K=K, N=N, M=M, L=L, side_info_ids=[0, 1])


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        PIRPSIClient(K=5, N=2, M=2, L=12, side_info_ids=[0, 1])
    with pytest.raises(ValueError):
        PIRPSIClient(K=5, N=4, M=2, L=12, side_info_ids=[0])


def test_wrong_number_of_servers_raises():
    N, K, M, L = 4, 5, 2, 12
    dataset = load_synthetic(K=K, L=L, seed=1)
    client, servers = _make_client(dataset, msg_id=0, M=M, N=N, L=L)
    with pytest.raises(ValueError):
        client.retrieve(0, servers[:3])
