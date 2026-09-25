"""Tests for the direct-access baseline."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.dataset import load_synthetic
from src.common.metrics import identity_bits, privacy_score
from src.direct.client import DirectClient
from src.direct.server import DirectServer


@pytest.fixture
def dataset():
    return load_synthetic(K=8, L=16, q=2, seed=1)


def test_server_returns_correct_message(dataset):
    server = DirectServer(dataset)
    msg = server.query(3)
    assert msg.msg_id == 3
    assert np.array_equal(msg.payload, dataset[3].payload)


def test_server_logs_the_query(dataset):
    server = DirectServer(dataset)
    server.query(5)
    assert server.log == [{"msg_id": 5}]
    assert any("msg_id" in entry and entry["msg_id"] == 5 for entry in server.log)


def test_server_raises_on_unknown_id(dataset):
    server = DirectServer(dataset)
    with pytest.raises(KeyError):
        server.query(99)


def test_server_len(dataset):
    assert len(DirectServer(dataset)) == 8


def test_client_retrieves_correct_message(dataset):
    server = DirectServer(dataset)
    client = DirectClient(server)
    msg = client.retrieve(2)
    assert msg.msg_id == 2
    assert np.array_equal(msg.payload, dataset[2].payload)


def test_privacy_leakage_is_full(dataset):
    """The server knows the exact msg_id -> privacy score must be 0."""
    server = DirectServer(dataset)
    K = len(dataset)
    server.query(0)
    leaked = identity_bits(K)  # log2(K) bits: which of K messages?
    assert leaked > 0
    assert privacy_score(leaked, identity_bits(K)) == 0.0