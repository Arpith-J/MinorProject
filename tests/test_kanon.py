"""Tests for the prefix-based k-anonymity module."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.dataset import load_synthetic
from src.kanon.bucket import (
    build_buckets,
    choose_bucket,
    compute_prefix,
    prefix_width,
)
from src.kanon.client import KAnonClient
from src.kanon.server import KAnonServer

K, L, PREFIX = 16, 16, 2


@pytest.fixture
def dataset():
    return load_synthetic(K=K, L=L, q=2, seed=5)


def test_build_buckets_sizes(dataset):
    assert len(dataset) == 16
    buckets, width = build_buckets(dataset, PREFIX, base=2)
    assert width == 4
    assert sorted(buckets) == ["00", "01", "10", "11"]
    for prefix, member_ids in buckets.items():
        assert len(member_ids) == 4  # 16 / 2**2


def test_choose_bucket(dataset):
    buckets, _ = build_buckets(dataset, PREFIX)
    prefix, member_ids = choose_bucket(5, buckets)
    assert prefix == "01"  # 5 -> "0101" in 4 bits -> prefix "01"
    assert 5 in member_ids
    assert len(member_ids) == 4


def test_compute_prefix_and_width():
    assert prefix_width(16, 2) == 4
    assert prefix_width(8, 2) == 3
    assert compute_prefix(5, 2, 2, 4) == "01"
    assert compute_prefix(0, 2, 2, 4) == "00"
    with pytest.raises(ValueError):
        compute_prefix(5, 0, 2, 4)


def test_choose_bucket_missing_id_raises(dataset):
    buckets, _ = build_buckets(dataset, PREFIX)
    with pytest.raises(ValueError):
        choose_bucket(99, buckets)


def test_client_retrieves_correct_message(dataset):
    server = KAnonServer(dataset, prefix_len=PREFIX, base=2)
    client = KAnonClient(server, known_ids=[m.msg_id for m in dataset], prefix_len=PREFIX)
    recovered = client.retrieve(9)
    assert recovered.msg_id == 9
    assert np.array_equal(recovered.payload, dataset[9].payload)


def test_server_learns_only_prefix(dataset):
    server = KAnonServer(dataset, prefix_len=PREFIX, base=2)
    server.query("10")
    assert len(server.log) == 1
    assert server.log[0] == {"prefix": "10"}
    assert not any("msg_id" in entry for entry in server.log)


def test_server_cannot_determine_exact_msg_id(dataset):
    """From a prefix alone the server faces multiple candidates."""
    server = KAnonServer(dataset, prefix_len=PREFIX, base=2)
    prefix, _ = choose_bucket(13, server.buckets)
    bucket = server.query(prefix)
    assert len(bucket) > 1


def test_privacy_level_is_bucket_size(dataset):
    server = KAnonServer(dataset, prefix_len=PREFIX, base=2)
    client = KAnonClient(server, known_ids=[m.msg_id for m in dataset], prefix_len=PREFIX)
    client.retrieve(6)
    assert client.privacy_level() == 4
    assert server.bucket_size(6) == 4


def test_client_rejects_unknown_id(dataset):
    server = KAnonServer(dataset, prefix_len=PREFIX)
    client = KAnonClient(server, known_ids=[m.msg_id for m in dataset], prefix_len=PREFIX)
    with pytest.raises(ValueError):
        client.retrieve(999)


def test_server_rejects_unknown_prefix(dataset):
    server = KAnonServer(dataset, prefix_len=PREFIX)
    with pytest.raises(KeyError):
        server.query("zz")


def test_privacy_level_requires_retrieval(dataset):
    server = KAnonServer(dataset, prefix_len=PREFIX)
    client = KAnonClient(server, known_ids=[m.msg_id for m in dataset], prefix_len=PREFIX)
    with pytest.raises(RuntimeError):
        client.privacy_level()