"""Tests for the comparison framework (runner and dashboard helpers)."""

from __future__ import annotations

import pytest

from src.comparison.dashboard import _build_config, build_report, plot_metrics, run_comparison
from src.comparison.runner import (
    default_config,
    load_config,
    prepare_dataset,
    run_all,
    run_pir_psi,
    summarize,
)
from src.common.dataset import load_synthetic


@pytest.fixture
def config():
    cfg = default_config()
    cfg.update({"K": 8, "N": 2, "M": 2, "L": 16, "q": 2, "random_seed": 42})
    return cfg


def test_run_all_returns_expected_structure(config):
    result = run_all(msg_id=3, config=config)
    assert set(result["approaches"]) == {"direct", "kanon", "pir", "pir_psi"}
    assert result["msg_id"] == 3
    for name, metrics in result["approaches"].items():
        assert "privacy" in metrics and "communication" in metrics
        assert "computation" in metrics and "latency_s" in metrics
        assert "recovered" in metrics
        assert metrics["recovered"] is True


def test_all_approaches_recover_bit_exact_message(config):
    dataset = prepare_dataset(config)
    result = run_all(msg_id=5, config=config)
    for name, metrics in result["approaches"].items():
        assert metrics["recovered"] is True
    assert len(dataset) == 8


def test_pir_has_highest_privacy(config):
    result = run_all(msg_id=2, config=config)
    privacy = {name: m["privacy"]["privacy_score"] for name, m in result["approaches"].items()}
    assert privacy["pir"] > privacy["kanon"] > privacy["direct"]
    assert privacy["direct"] == 0.0
    assert privacy["pir"] == 1.0
    assert privacy["pir_psi"] == 1.0


def test_pir_psi_rate_beats_chor_for_positive_side_info(config):
    # config has M=2 > 0; PIR-PSI should beat the Chor baseline rate 1/2.
    result = run_all(msg_id=2, config=config)
    psi = result["approaches"]["pir_psi"]
    assert psi["capacity_rate"] > 0.5
    assert psi["rate"] > 0.5
    assert psi["servers"] >= config["M"] + 1


def test_direct_has_lowest_latency_kanon_between(config):
    result = run_all(msg_id=2, config=config)
    lat = {name: m["latency_s"] for name, m in result["approaches"].items()}
    assert lat["direct"] <= lat["kanon"] <= lat["pir"]


def test_communication_ordering(config):
    result = run_all(msg_id=2, config=config)
    comm = {name: m["communication"]["total_bytes"] for name, m in result["approaches"].items()}
    assert comm["kanon"] > comm["direct"]  # bucket is larger than a single message
    assert comm["pir"] > comm["direct"]


def test_summarize_and_dashboard_helpers(config):
    result = run_all(msg_id=4, config=config)
    report = build_report(result)
    assert list(report.columns) == [
        "approach", "privacy_score", "leakage_bits", "bytes_sent",
        "bytes_received", "total_bytes", "latency_s",
    ]
    assert len(report) == 4
    fig = plot_metrics(report)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_summarize_equals_direct_call(config):
    result = run_all(msg_id=0, config=config)
    assert summarize(result) is not None
    assert summarize(result).shape == (4, 7)


def test_dashboard_build_config():
    cfg = _build_config(K=8, N=2, M=2, L=16, q=2, seed=1, prefix_len=2, latency_repeats=3)
    assert cfg["K"] == 8 and cfg["kanon"]["prefix_len"] == 2
    assert cfg["M"] == 2 and cfg["random_seed"] == 1
    assert cfg["pir_psi"]["enabled"] is True


def test_dashboard_run_comparison(config):
    result = run_comparison(config, msg_id=1)
    assert set(result["approaches"]) == {"direct", "kanon", "pir", "pir_psi"}


def test_run_pir_psi_public_api(config):
    result = run_pir_psi(msg_id=4, config=config)
    assert result["recovered"] is True
    assert result["privacy"]["privacy_score"] == 1.0
    assert result["sub_packetization"] == result["servers"] - 1
    assert result["capacity_rate"] > 0.5
    assert 0.0 <= result["rate_ratio"] <= 1.5


def test_load_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("K: 8\nN: 2\nM: 1\nL: 16\nq: 2\ndataset_path: null\nrandom_seed: 3\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg["K"] == 8 and cfg["random_seed"] == 3
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_run_all_invalid_msg_id(config):
    with pytest.raises(ValueError):
        run_all(msg_id=99, config=config)


def test_dataset_reproducible(config):
    a = prepare_dataset(config)
    b = prepare_dataset(config)
    for ma, mb in zip(a, b):
        assert (ma.payload == mb.payload).all()
    assert load_synthetic(K=4, L=8, seed=1) is not None