"""Tests for PIR-PSI capacity / rate (Sub-phase 8d)."""

from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest

from src.common.dataset import load_synthetic
from src.pir_psi.capacity import (
    capacity_rate,
    expected_download_bits,
    is_achievable,
    measure_actual_rate,
)
from src.pir_psi.client import PIRPSIClient
from src.pir_psi.server import PIRPSIServer


class TestCapacityRate:
    def test_erhili_example(self):
        # (N=4, K=5, M=2) -> 16/21, the paper's worked example.
        assert capacity_rate(4, 5, 2) == pytest.approx(16 / 21)
        assert Fraction(48, 63) == Fraction(16, 21)

    def test_chor_baseline_m_zero(self):
        # Finite-length PIR capacity for N=2, K=4, M=0 is the Sun-Jafar
        # value 8/15; its K -> infinity limit is the Chor rate (N-1)/N = 1/2.
        assert capacity_rate(2, 4, 0) == pytest.approx(8 / 15)
        assert capacity_rate(2, 30, 0) == pytest.approx(0.5, abs=1e-6)

    def test_rate_increases_with_side_information(self):
        base = capacity_rate(3, 8, 0)
        assert capacity_rate(3, 8, 1) > base
        assert capacity_rate(3, 8, 2) > capacity_rate(3, 8, 1)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            capacity_rate(2, 5, 2)  # N < M+1
        with pytest.raises(ValueError):
            capacity_rate(4, 2, 2)  # K <= M


class TestAchievability:
    def test_accepts_valid(self):
        assert is_achievable(4, 5, 2, 12)
        assert is_achievable(2, 4, 0, 8)

    def test_rejects_indivisible_length(self):
        # L=10, N-1=3 -> 10 % 3 != 0.
        assert not is_achievable(4, 5, 2, 10)

    def test_rejects_bad_server_count(self):
        assert not is_achievable(2, 5, 2, 12)


class TestExpectedDownload:
    def test_matches_formula(self):
        assert expected_download_bits(4, 5, 2, 12) == pytest.approx(12 / (16 / 21))


def test_measure_actual_rate_converges_to_capacity():
    N, K, M, L = 4, 5, 2, 12
    dataset = load_synthetic(K=K, L=L, seed=42)
    servers = [PIRPSIServer(i, dataset, N=N) for i in range(N)]
    cap = capacity_rate(N, K, M)
    rng = random.Random(0)
    total = 0.0
    trials = 1000
    for _ in range(trials):
        msg_id = rng.randrange(K)
        side = rng.sample([i for i in range(K) if i != msg_id], M)
        client = PIRPSIClient(
            K=K, N=N, M=M, L=L, q=2,
            side_info_ids=side, side_info_messages=[dataset[i] for i in side],
            seed=rng.getrandbits(16),
        )
        total += measure_actual_rate(client, msg_id, servers)
    mean_rate = total / trials
    assert abs(mean_rate - cap) / cap < 0.05


def test_measure_actual_rate_single_call_range():
    N, K, M, L = 4, 5, 2, 12
    dataset = load_synthetic(K=K, L=L, seed=5)
    servers = [PIRPSIServer(i, dataset, N=N) for i in range(N)]
    client = PIRPSIClient(
        K=K, N=N, M=M, L=L, q=2, side_info_ids=[0, 1],
        side_info_messages=[dataset[0], dataset[1]], seed=1,
    )
    rate = measure_actual_rate(client, 2, servers)
    assert 0.5 <= rate <= 1.0
