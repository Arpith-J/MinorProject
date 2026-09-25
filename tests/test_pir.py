"""Tests for the two-server PIR module (query generation, servers, client)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.common.dataset import load_synthetic
from src.pir.client import PIRClient
from src.pir.query import generate_query_vectors
from src.pir.server import PIRServer


@pytest.fixture
def dataset():
    return load_synthetic(K=4, L=8, q=2, seed=11)


@pytest.fixture
def servers(dataset):
    return [PIRServer(dataset, server_id=0), PIRServer(dataset, server_id=1)]


class TestQueryGeneration:
    def test_queries_shapes_and_binary(self):
        queries = []
        for _ in range(400):
            rng = random.Random(0)
            queries = generate_query_vectors(msg_id=2, side_info_ids=[], N=2, K=4, rng=rng)
            assert len(queries) == 2
            for q in queries:
                assert q.shape == (4,)
                assert set(np.unique(q)).issubset({0, 1})
        assert len([1 for q in queries if q.any()]) == 2

    def test_queries_combine_to_desired_position(self):
        rng = random.Random(0)
        q0, q1 = generate_query_vectors(msg_id=2, side_info_ids=[], N=2, K=4, rng=rng)
        assert np.array_equal(np.bitwise_xor(q0, q1), np.asarray([0, 0, 1, 0], dtype=np.int8))

    def test_side_info_excluded_from_both_queries(self):
        rng = random.Random(1)
        q0, q1 = generate_query_vectors(msg_id=2, side_info_ids=[0, 3], N=2, K=4, rng=rng)
        assert q0[0] == 0 and q0[3] == 0
        assert q1[0] == 0 and q1[3] == 0

    def test_invalid_parameters_raise(self):
        rng = random.Random(0)
        with pytest.raises(ValueError):
            generate_query_vectors(4, [], 2, 4, rng)  # msg_id out of range
        with pytest.raises(ValueError):
            generate_query_vectors(1, [1], 2, 4, rng)  # msg_id in side info
        with pytest.raises(ValueError):
            generate_query_vectors(1, [1, 2, 3], 2, 4, rng)  # only 1 free id
        with pytest.raises(ValueError):
            generate_query_vectors(1, [], 2, 0, rng)  # empty dataset

    def test_reproducible_under_same_seed(self):
        a = generate_query_vectors(0, [], 2, 4, random.Random(7))
        b = generate_query_vectors(0, [], 2, 4, random.Random(7))
        for qa, qb in zip(a, b):
            assert np.array_equal(qa, qb)

    def test_server0_cannot_distinguish_msg_id(self):
        """Server 0's view must be identically uniform for every msg_id."""
        K, trials = 4, 2000
        base = None
        for msg_id in range(K):
            rng = random.Random(1234)
            mean = np.zeros(K)
            for _ in range(trials):
                q0, _ = generate_query_vectors(msg_id, [], 2, K, rng)
                mean += q0
            mean /= trials
            assert np.all(np.abs(mean - 0.5) < 0.04)  # uniform marginals
            if base is None:
                base = mean
            else:
                assert np.max(np.abs(mean - base)) < 0.05

    def test_server1_cannot_distinguish_msg_id(self):
        """Server 1's view must be identically uniform for every msg_id."""
        K, trials = 4, 2000
        base = None
        for msg_id in range(K):
            rng = random.Random(999)
            mean = np.zeros(K)
            for _ in range(trials):
                _, q1 = generate_query_vectors(msg_id, [], 2, K, rng)
                mean += q1
            mean /= trials
            assert np.all(np.abs(mean - 0.5) < 0.04)
            if base is None:
                base = mean
            else:
                assert np.max(np.abs(mean - base)) < 0.05

    def test_n_servers_generates_n_vectors(self):
        rng = random.Random(2)
        queries = generate_query_vectors(1, [0], 3, 4, rng)
        assert len(queries) == 3
        q0, q1, qd = queries
        assert np.array_equal(np.bitwise_xor(q0, q1), np.asarray([0, 1, 0, 0], dtype=np.int8))
        assert qd.shape == (4,)


class TestPIRServer:
    def test_answers_are_valid_xor_sums(self, dataset, servers):
        q0, q1 = generate_query_vectors(2, [], 2, 4, random.Random(0))
        a0 = servers[0].answer(q0)
        a1 = servers[1].answer(q1)
        assert a0 is not None and a1 is not None
        assert a0.shape == (8,) and a1.shape == (8,)
        assert np.array_equal(np.bitwise_xor(a0, a1), dataset[2].payload)

    def test_zero_query_returns_none(self, dataset):
        server = PIRServer(dataset)
        assert server.answer(np.zeros(4, dtype=np.int8)) is None
        assert server.answer_zero() is None

    def test_wrong_length_query_raises(self, dataset):
        server = PIRServer(dataset)
        with pytest.raises(ValueError):
            server.answer(np.zeros(3, dtype=np.int8))

    def test_servers_log_queries(self, servers):
        q0, q1 = generate_query_vectors(1, [], 2, 4, random.Random(5))
        servers[0].answer(q0)
        servers[1].answer(q1)
        assert servers[0].log and "query" in servers[0].log[0]


class TestPIRClient:
    def test_client_recovers_correct_message(self, dataset, servers):
        client = PIRClient(servers, seed=3)
        recovered = client.retrieve(2, [])
        assert recovered.msg_id == 2
        assert np.array_equal(recovered.payload, dataset[2].payload)

    def test_recovered_message_bit_exact(self, dataset, servers):
        client = PIRClient(servers, seed=3)
        for msg_id in range(4):
            recovered = client.retrieve(msg_id, [])
            assert np.array_equal(recovered.payload, dataset[msg_id].payload)

    def test_100_random_trials_recover_everything(self):
        K, L = 6, 16
        dataset = load_synthetic(K=K, L=L, seed=42)
        servers = [PIRServer(dataset, 0), PIRServer(dataset, 1)]
        client = PIRClient(servers, seed=42)
        rng = random.Random(0)
        for trial in range(100):
            msg_id = rng.randrange(K)
            pool = [i for i in range(K) if i != msg_id]
            rng.shuffle(pool)
            side = pool[:2]  # M = 2 side-info messages
            recovered = client.retrieve(msg_id, side)
            assert np.array_equal(recovered.payload, dataset[msg_id].payload), (
                f"trial {trial} failed for msg_id {msg_id}"
            )

    def test_with_three_servers(self, dataset):
        servers = [
            PIRServer(dataset, 0),
            PIRServer(dataset, 1),
            PIRServer(dataset, 2),
        ]
        client = PIRClient(servers, seed=1)
        recovered = client.retrieve(2, [])
        assert np.array_equal(recovered.payload, dataset[2].payload)

    def test_privacy_leakage_is_zero(self, servers):
        client = PIRClient(servers)
        assert client.privacy_leakage_bits() == 0.0

    def test_stats_and_communication(self, servers):
        client = PIRClient(servers, seed=9)
        client.retrieve(0, [])
        stats = client.last_communication()
        assert stats["bytes_received"] > 0
        assert stats["total_bytes"] == stats["bytes_sent"] + stats["bytes_received"]
        with pytest.raises(RuntimeError):
            PIRClient(servers).last_communication()

    def test_client_requires_two_servers(self, dataset):
        with pytest.raises(ValueError):
            PIRClient([PIRServer(dataset, 0)])