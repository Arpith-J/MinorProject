"""Tests for the PIR-PSI query-generation module (Sub-phase 8a)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.pir_psi.query import (
    build_query,
    combinatorial_m,
    generate_query_vectors,
    permute_vectors,
    probability_matrix,
    sample_query_parameters,
)


class TestCombinatorialM:
    def test_base_case(self):
        assert combinatorial_m(0, 0, 2, 4) == 1

    def test_zero_region(self):
        M = 3
        for i in range(M + 1):
            for j in range(M + 1):
                if 1 <= i + j <= M:
                    assert combinatorial_m(i, j, M, 5) == 0

    def test_known_values(self):
        # m_{i,j} = N-1 for all i+j == M+1 (paper, proof of Lemma 2).
        assert combinatorial_m(1, 2, 2, 4) == 3  # N-1 = 3
        assert combinatorial_m(2, 1, 2, 4) == 3
        assert combinatorial_m(0, 3, 2, 4) == 3
        assert combinatorial_m(2, 2, 2, 4) == 3


class TestProbabilityMatrix:
    def test_sums_to_one(self):
        P = probability_matrix(M=2, K=5, N=4)
        assert P.shape == (3, 3)
        assert np.isclose(P.sum(), 1.0, atol=1e-9)

    def test_nonnegative(self):
        P = probability_matrix(M=2, K=5, N=4)
        assert np.all(P >= -1e-12)

    def test_known_entries_for_erhili_example(self):
        P = probability_matrix(M=2, K=5, N=4)
        assert np.isclose(P[0, 0], 0.0625)
        assert np.isclose(P[1, 2], 0.375)
        assert np.isclose(P[2, 1], 0.375)
        assert np.isclose(P[2, 2], 0.1875)
        assert np.isclose(P[0, 1], 0.0)

    def test_m_zero_distribution(self):
        # For M=0 the matrix is 1 x K and must sum to 1; for N=2 it equals
        # the Binomial(K-1, 1/2) distribution over J.
        K, N = 5, 2
        P = probability_matrix(M=0, K=K, N=N)
        assert P.shape == (1, K)
        assert np.isclose(P.sum(), 1.0)
        from math import comb

        for j in range(K):
            assert np.isclose(P[0, j], comb(K - 1, j) / 2 ** (K - 1))

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            probability_matrix(M=3, K=5, N=3)  # N < M+1
        with pytest.raises(ValueError):
            probability_matrix(M=5, K=5, N=6)  # K <= M


class TestSampleParameters:
    def test_valid_range_over_many_trials(self):
        M, K, N = 2, 5, 4
        rng = random.Random(0)
        for _ in range(10_000):
            I, J = sample_query_parameters(M, K, N, rng)
            assert 0 <= I <= M
            assert 0 <= J <= K - M - 1

    def test_empirical_matches_theoretical(self):
        M, K, N = 2, 5, 4
        rng = random.Random(123)
        trials = 10_000
        counts = np.zeros((M + 1, K - M))
        for _ in range(trials):
            I, J = sample_query_parameters(M, K, N, rng)
            counts[I, J] += 1
        empirical = counts / trials
        theoretical = probability_matrix(M, K, N)
        assert np.max(np.abs(empirical - theoretical)) < 0.02


class TestGenerateQueryVectors:
    def test_shapes_and_ranges(self):
        K, N, M = 5, 4, 2
        vectors = generate_query_vectors(3, [0, 1], K, N, M, random.Random(7))
        assert len(vectors) == N
        for v in vectors:
            assert v.shape == (K,)
            assert np.all(v >= 0) and np.all(v <= N - 1)

    def test_demand_entry_in_n_minus_1_vectors(self):
        K, N, M = 5, 4, 2
        rng = random.Random(1)
        vectors = generate_query_vectors(3, [0, 1], K, N, M, rng)
        with_demand = sum(1 for v in vectors if v[3] != 0)
        assert with_demand == N - 1

    def test_demand_subpackets_form_a_permutation(self):
        K, N, M = 5, 4, 2
        plan = build_query(3, [0, 1], K, N, M, random.Random(5))
        # The N-1 demand sub-packets recovered from the N-1 "a" vectors are
        # exactly the sub-packet indices 1 .. N-1, each used once.
        demand_values = sorted(int(v[3]) for v in plan.vectors if v[3] != 0)
        assert demand_values == list(range(1, N))

    def test_reproducible_under_same_seed(self):
        a = generate_query_vectors(2, [0, 1], 5, 4, 2, random.Random(9))
        b = generate_query_vectors(2, [0, 1], 5, 4, 2, random.Random(9))
        for va, vb in zip(a, b):
            assert np.array_equal(va, vb)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            generate_query_vectors(5, [0, 1], 5, 4, 2, random.Random(0))  # W range
        with pytest.raises(ValueError):
            generate_query_vectors(2, [0], 5, 4, 2, random.Random(0))  # |S| != M
        with pytest.raises(ValueError):
            generate_query_vectors(2, [2], 5, 4, 2, random.Random(0))  # W in S
        with pytest.raises(ValueError):
            generate_query_vectors(2, [0, 1], 5, 2, 2, random.Random(0))  # N<M+1


class TestPermuteVectors:
    def test_permutation_preserves_multiset(self):
        vectors = [np.array([i, i + 1]) for i in range(4)]
        out = permute_vectors(vectors, random.Random(3))
        assert len(out) == 4
        assert sorted(tuple(v) for v in out) == sorted(tuple(v) for v in vectors)

    def test_is_a_permutation(self):
        vectors = [np.array([i]) for i in range(5)]
        out = permute_vectors(vectors, random.Random(4))
        assert sorted(int(v[0]) for v in out) == list(range(5))
