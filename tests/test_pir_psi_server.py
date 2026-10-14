"""Tests for the PIR-PSI server (Sub-phase 8b)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.message import Message
from src.pir_psi.server import PIRPSIServer


def _tiny_dataset():
    # 4 messages of length 3, split into N-1 = 3 sub-packets of length 1.
    payloads = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
    ]
    return [Message(i, np.asarray(p, dtype=np.uint8), q=2) for i, p in enumerate(payloads)]


class TestPIRPSIServer:
    def test_all_zero_query_returns_none(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        assert server.answer(np.zeros(4, dtype=np.int64)) is None

    def test_hand_example_xor(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        # v selects msg0 sub1, msg1 sub1, msg2 sub1, msg3 sub1:
        #   1 ^ 0 ^ 1 ^ 0 = 0
        assert np.array_equal(server.answer(np.array([1, 1, 1, 1])), np.array([0]))

    def test_hand_example_xor_second(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        # msg0 sub3=1, msg1 sub1=0, msg2 sub2=1, msg3 sub3=1 -> 1
        assert np.array_equal(server.answer(np.array([3, 1, 2, 3])), np.array([1]))

    def test_degenerate_zero_subpackets_ignored(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        a = server.answer(np.array([1, 0, 0, 0]))
        # msg0 sub1 = 1
        assert np.array_equal(a, np.array([1]))
        b = server.answer(np.array([1, 0, 3, 0]))
        # msg0 sub1=1 XOR msg2 sub3=0 -> 1
        assert np.array_equal(b, np.array([1]))

    def test_multiple_subpackets_per_message(self):
        payloads = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0]]
        messages = [
            Message(i, np.asarray(p, dtype=np.uint8), q=2)
            for i, p in enumerate(payloads)
        ]
        server = PIRPSIServer(0, messages, N=4)
        # msg0 sub1=1, msg1 sub2=0, msg2 sub3=1, msg3 sub1=1 -> 1
        assert np.array_equal(server.answer(np.array([1, 2, 3, 1])), np.array([1]))

    def test_query_reuse_consistent(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        q = np.array([3, 2, 1, 2])
        first = server.answer(q)
        second = server.answer(q)
        assert np.array_equal(first, second)

    def test_answer_size(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        assert server.answer_size_bytes() == 1
        assert server.answer_size_bits() == 1

    def test_wrong_length_query_raises(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        with pytest.raises(ValueError):
            server.answer(np.zeros(3, dtype=np.int64))

    def test_out_of_range_entry_raises(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        with pytest.raises(ValueError):
            server.answer(np.array([0, 0, 4, 0]))

    def test_indivisible_length_raises(self):
        messages = [Message(0, np.asarray([1, 1], dtype=np.uint8))]
        with pytest.raises(ValueError):
            PIRPSIServer(0, messages, N=4)  # 2 not divisible by 3

    def test_server_only_records_query(self):
        server = PIRPSIServer(0, _tiny_dataset(), N=4)
        server.answer(np.array([1, 0, 0, 2]))
        entry = server.log[0]
        assert set(entry.keys()) == {"server_id", "query"}
        # The server has no attribute that stores a demand or side-info id.
        assert not hasattr(server, "demand")
        assert not hasattr(server, "side_info_ids")

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            PIRPSIServer(0, _tiny_dataset(), N=1)
        with pytest.raises(ValueError):
            PIRPSIServer(0, [], N=4)
