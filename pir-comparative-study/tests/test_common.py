"""Tests for the common utilities (message, dataset, metrics)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.dataset import get_ground_truth, load_csv, load_synthetic
from src.common.message import Message
from src.common.metrics import (
    communication_cost,
    computation_cost,
    identity_bits,
    latency,
    privacy_score,
)


class TestMessage:
    def test_synthetic_messages_have_expected_length(self):
        messages = load_synthetic(K=5, L=32, q=2, seed=7)
        assert len(messages) == 5
        for msg in messages:
            assert msg.length == 32
            assert msg.payload.shape == (32,)
            assert set(np.unique(msg.payload)).issubset({0, 1})

    def test_split_subpackets_equal_length(self):
        # L=36 so that L is divisible by N-1=3 sub-packets.
        messages = load_synthetic(K=5, L=36, q=2, seed=7)
        n = 3  # N - 1 sub-packets
        for msg in messages:
            sub = msg.split_subpackets(n)
            assert len(sub) == n
            for sp in sub:
                assert sp.length == 36 // n  # L / (N - 1)
                assert sp.msg_id == msg.msg_id
                assert np.array_equal(
                    np.concatenate([s.payload for s in sub]), msg.payload
                )

    def test_split_subpackets_requires_divisibility(self):
        msg = Message(0, np.zeros(7, dtype=np.uint8))
        with pytest.raises(ValueError):
            msg.split_subpackets(3)

    def test_message_validates_payload_range(self):
        with pytest.raises(ValueError):
            Message(0, np.asarray([0, 2], dtype=np.uint8), q=2)

    def test_to_binary_for_q2(self):
        msg = Message(0, np.asarray([1, 0, 1, 1], dtype=np.uint8), q=2)
        out = msg.to_binary()
        assert np.array_equal(out, msg.payload)
        assert out is not msg.payload  # returned a fresh copy

    def test_to_binary_rejects_q_gt_2(self):
        with pytest.raises(ValueError):
            Message(0, np.asarray([0, 1, 2], dtype=np.uint8), q=3).to_binary()

    def test_bytes_roundtrip(self):
        msg = Message(3, np.asarray([1, 0, 1, 0, 1, 1], dtype=np.uint8))
        restored = Message.from_bytes(3, msg.to_bytes())
        assert restored.msg_id == msg.msg_id
        assert np.array_equal(restored.payload, msg.payload)


class TestDataset:
    def test_synthetic_is_reproducible(self):
        a = load_synthetic(K=8, L=16, q=2, seed=42)
        b = load_synthetic(K=8, L=16, q=2, seed=42)
        for ma, mb in zip(a, b):
            assert np.array_equal(ma.payload, mb.payload)

    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "ds.csv"
        csv_file.write_text("id,payload\n0,0,1,0\n1,1,0,1\n", encoding="utf-8")
        messages = load_csv(csv_file)
        assert len(messages) == 2
        assert messages[0].msg_id == 0
        assert np.array_equal(messages[0].payload, np.asarray([0, 1, 0], dtype=np.uint8))

    def test_load_csv_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nope.csv")

    def test_load_csv_rejects_bad_rows(self, tmp_path):
        bad_payload = tmp_path / "bad.csv"
        bad_payload.write_text("id,payload\n0\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_csv(bad_payload)

    def test_ground_truth_lookup(self):
        dataset = load_synthetic(K=8, L=16, seed=3)
        gt = get_ground_truth(4, dataset)
        assert gt.msg_id == 4
        assert gt.length == 16

    def test_ground_truth_missing_id(self):
        with pytest.raises(ValueError):
            get_ground_truth(99, load_synthetic(K=4, L=8))


class TestMetrics:
    def test_privacy_score_types_and_ranges(self):
        scores = {
            "direct": privacy_score(identity_bits(8), identity_bits(8)),
            "kanon": privacy_score(math.log2(4), identity_bits(8)),
            "pir": privacy_score(0.0, identity_bits(8)),
        }
        for name, score in scores.items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        assert scores["direct"] == 0.0
        assert scores["kanon"] == pytest.approx(1.0 - 2.0 / 3.0)  # 1 - log2(4)/log2(8)
        assert scores["pir"] == 1.0

    def test_privacy_score_edges(self):
        assert privacy_score(3.0, 0.0) == 1.0  # no identity space -> perfect
        assert privacy_score(100.0, 3.0) == 0.0  # clamping

    def test_communication_cost(self):
        cost = communication_cost(4, 100)
        assert cost == {"bytes_sent": 4.0, "bytes_received": 100.0, "total_bytes": 104.0}

    def test_computation_cost(self):
        cost = computation_cost(10, "xors")
        assert cost == {"op_type": "xors", "num_ops": 10}

    def test_latency(self):
        assert latency(1.0, 3.5) == pytest.approx(2.5)
        assert latency(3.5, 1.0) == 0.0  # clamped to zero