"""Streamlit comparison dashboard.

Launch with::

    streamlit run src/comparison/dashboard.py

The pure report/plot helpers (``build_report``, ``run_comparison``,
``plot_metrics``) do not touch Streamlit and are unit-tested.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.comparison.runner import run_all, summarize  # noqa: E402

APPROACHES = ("direct", "kanon", "pir")


def run_comparison(
    config: Dict[str, object], msg_id: int, side_info_ids: Optional[List[int]] = None
) -> Dict[str, object]:
    """Run all approaches and return the raw structured result."""
    return run_all(msg_id, side_info_ids, config)


def build_report(result: Dict[str, object]) -> pd.DataFrame:
    """Flatten a run result into a comparison table."""
    return summarize(result)


def plot_metrics(report: pd.DataFrame) -> plt.Figure:
    """Build a three-panel bar chart (privacy, latency, communication).

    Args:
        report: DataFrame from :func:`build_report`.

    Returns:
        A matplotlib figure with privacy, latency and bytes panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    order = sorted({*APPROACHES, *report["approach"].unique()})
    colors = {"direct": "#c0392b", "kanon": "#e67e22", "pir": "#27ae60"}
    palette = [colors.get(a, "#7f8c8d") for a in report["approach"]]

    axes[0].bar(report["approach"], report["privacy_score"], color=palette)
    axes[0].set_title("Privacy score (higher = better)")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(report["approach"], report["latency_s"] * 1e6, color=palette)
    axes[1].set_title("Latency (microseconds)")
    axes[1].tick_params(axis="x", labelrotation=30)

    axes[2].bar(report["approach"], report["total_bytes"], color=palette)
    axes[2].set_title("Total bytes exchanged")
    axes[2].tick_params(axis="x", labelrotation=30)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _build_config(
    K: int, N: int, M: int, L: int, q: int, seed: int, prefix_len: int,
    latency_repeats: int,
) -> Dict[str, object]:
    """Assemble a configuration dict from the sidebar values."""
    return {
        "N": N,
        "K": K,
        "M": M,
        "L": L,
        "q": q,
        "dataset_path": None,
        "random_seed": seed,
        "latency_repeats": latency_repeats,
        "kanon": {"prefix_len": prefix_len, "base": 2},
    }


def main() -> None:
    """Streamlit entry point (executed by `streamlit run ...`)."""
    import streamlit as st

    st.set_page_config(page_title="PIR Comparative Study", layout="wide")
    st.title("Comparative Study of Privacy-Preserving Information Retrieval")
    st.caption(
        "Direct access vs prefix-based k-anonymity vs two-server PIR "
        "on the same synthetic dataset."
    )

    with st.sidebar:
        st.header("Dataset & protocol parameters")
        K = st.slider("K (number of messages)", 4, 32, 8)
        N = st.selectbox("N (PIR servers)", [2, 3], index=0)
        M = st.slider("M (side information)", 0, max(0, K - 2), min(2, max(0, K - 2)))
        L = st.slider("L (message length)", 8, 128, 16, step=8)
        q = st.selectbox("q (field size)", [2], index=0)
        prefix_len = st.slider("K-anonymity prefix length", 1, max(1, K.bit_length()), 1)
        latency_repeats = st.slider("Latency repeats", 1, 20, 10)
        seed = st.number_input("Random seed", 0, 99999, 42, step=1)
        msg_id = st.selectbox("Message to retrieve", list(range(K)), index=K // 2)
        approach = st.radio("Approach", ["All", "Direct", "K-Anonymity", "PIR"], horizontal=True)
        run = st.button("Run comparison", type="primary", use_container_width=True)

    if not run:
        st.info("Tune the parameters on the left and press **Run comparison**.")
        return

    config = _build_config(K, N, M, L, q, int(seed), prefix_len, latency_repeats)
    result = run_all(msg_id, None, config)
    report = build_report(result)

    st.subheader(f"Retrieving message #{msg_id}  (K={K}, N={N}, M={M}, L={L})")
    col_table, col_plot = st.columns([1, 2])
    with col_table:
        st.dataframe(report, use_container_width=True, hide_index=True)
    with col_plot:
        st.pyplot(plot_metrics(report), use_container_width=True)

    st.subheader("Detailed metrics")
    cols = st.columns(3)
    for col, name in zip(cols, APPROACHES):
        with col:
            metrics = result["approaches"][name]
            st.markdown(f"**{name.title()}** — recovered: "
                        f"{'OK' if metrics['recovered'] else 'MISMATCH'}")
            st.json(
                {
                    "privacy_score": metrics["privacy"]["privacy_score"],
                    "leakage_bits": metrics["privacy"]["leakage_bits"],
                    "bytes_sent": metrics["communication"]["bytes_sent"],
                    "bytes_received": metrics["communication"]["bytes_received"],
                    "computation": metrics["computation"],
                    "latency_s": round(metrics["latency_s"], 6),
                }
            )


if __name__ == "__main__":
    main()