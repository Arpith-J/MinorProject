"""Parameter sweeps for the comparative study.

Sweeps ``K in [4, 8, 16]``, ``N in [2, 3]`` and ``M in [0, 2]`` over the
comparison runner, then writes:

* ``docs/results.csv``  -- one row per (config, approach)
* ``docs/results.md``   -- tables + embedded plots
* ``docs/plots/*.png``  -- privacy / communication / latency / rate vs K

Usage::

    python scripts/run_experiments.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.comparison.runner import run_all  # noqa: E402

APPROACHES = ("direct", "kanon", "pir")

KVALS = [4, 8, 16]
NVALS = [2, 3]
M_OPTIONS = [0, 2]
L = 16
SEED = 42
PREFIX_LEN = 1  # fixed prefix so bucket sizes scale with K


def build_config(K: int, N: int, M: int, prefix_len: int) -> dict:
    return {
        "N": N,
        "K": K,
        "M": M,
        "L": L,
        "q": 2,
        "dataset_path": None,
        "random_seed": SEED,
        "latency_repeats": 3,
        "kanon": {"prefix_len": prefix_len, "base": 2},
    }


def run_sweep() -> pd.DataFrame:
    rows = []
    for K in KVALS:
        for N in NVALS:
            for M in M_OPTIONS:
                if K - M < 2:  # PIR needs >= 2 non-side-info messages
                    continue
                cfg = build_config(K, N, M, PREFIX_LEN)
                result = run_all(msg_id=K // 2, side_info_ids=None, config=cfg)
                for approach, metrics in result["approaches"].items():
                    rows.append(
                        {
                            "K": K,
                            "N": N,
                            "M": M,
                            "approach": approach,
                            "privacy_score": metrics["privacy"]["privacy_score"],
                            "leakage_bits": metrics["privacy"]["leakage_bits"],
                            "bytes_sent": metrics["communication"]["bytes_sent"],
                            "bytes_received": metrics["communication"]["bytes_received"],
                            "total_bytes": metrics["communication"]["total_bytes"],
                            "latency_s": metrics["latency_s"],
                            "recovered": metrics["recovered"],
                            "rate": L / metrics["communication"]["total_bytes"],
                        }
                    )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate privacy, communication, latency and rate plots vs K."""
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["K", "approach"], as_index=False)[
        ["privacy_score", "total_bytes", "latency_s", "rate"]
    ].mean()

    # 1) privacy vs K
    fig, ax = plt.subplots(figsize=(6, 4))
    for approach, color in zip(APPROACHES, ["#c0392b", "#e67e22", "#27ae60"]):
        sub = grouped[grouped["approach"] == approach]
        ax.plot(sub["K"], sub["privacy_score"], marker="o", color=color, label=approach)
    ax.set_xlabel("K (messages)")
    ax.set_ylabel("Privacy score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Privacy vs K")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "privacy_vs_K.png", dpi=120)
    plt.close(fig)

    # 2) communication vs K
    fig, ax = plt.subplots(figsize=(6, 4))
    for approach, color in zip(APPROACHES, ["#c0392b", "#e67e22", "#27ae60"]):
        sub = grouped[grouped["approach"] == approach]
        ax.plot(sub["K"], sub["total_bytes"], marker="s", color=color, label=approach)
    ax.set_xlabel("K (messages)")
    ax.set_ylabel("Total bytes exchanged")
    ax.set_title("Communication vs K")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "communication_vs_K.png", dpi=120)
    plt.close(fig)

    # 3) latency vs K
    fig, ax = plt.subplots(figsize=(6, 4))
    for approach, color in zip(APPROACHES, ["#c0392b", "#e67e22", "#27ae60"]):
        sub = grouped[grouped["approach"] == approach]
        ax.plot(sub["K"], sub["latency_s"] * 1e6, marker="^", color=color, label=approach)
    ax.set_xlabel("K (messages)")
    ax.set_ylabel("Latency (us)")
    ax.set_title("Latency vs K")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_K.png", dpi=120)
    plt.close(fig)

    # 4) retrieval rate vs K (message bits per exchanged byte)
    fig, ax = plt.subplots(figsize=(6, 4))
    for approach, color in zip(APPROACHES, ["#c0392b", "#e67e22", "#27ae60"]):
        sub = grouped[grouped["approach"] == approach]
        ax.plot(sub["K"], sub["rate"], marker="x", color=color, label=approach)
    ax.set_xlabel("K (messages)")
    ax.set_ylabel("Rate (bits / byte)")
    ax.set_title("Retrieval rate vs K")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rate_vs_K.png", dpi=120)
    plt.close(fig)


def write_markdown(df: pd.DataFrame, md_path: Path, plots_dir: Path) -> None:
    """Render the results table and plot references into results.md."""
    lines = [
        "# Experiment Results",
        "",
        "Auto-generated by `scripts/run_experiments.py` "
        f"(L={L}, q=2, seed={SEED}).",
        "",
        "## Approach summary",
        "",
        "| K | N | M | approach | privacy | leaked bits | sent (B) | recv (B) | total (B) | latency (us) | recovered |",
        "|---|:--:|:--:|----------|--------:|------------:|---------:|---------:|----------:|-------------:|:---------:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {int(row.K)} | {int(row.N)} | {int(row.M)} | {row.approach} "
            f"| {row.privacy_score:.2f} | {row.leakage_bits:.2f} "
            f"| {row.bytes_sent:.0f} | {row.bytes_received:.0f} | {row.total_bytes:.0f} "
            f"| {row.latency_s * 1e6:.1f} | {str(row.recovered).lower()} |"
        )
    lines += [
        "",
        "## Plots",
        "",
        "![Privacy vs K](plots/privacy_vs_K.png)",
        "",
        "![Communication vs K](plots/communication_vs_K.png)",
        "",
        "![Latency vs K](plots/latency_vs_K.png)",
        "",
        "![Retrieval rate vs K](plots/rate_vs_K.png)",
        "",
        "## Privacy ordering",
        "",
        f"* Direct access leaks {math.log2(4):.0f}-bit ids to {math.log2(16):.0f} bits "
        "(privacy score 0).",
        "* K-Anonymity leaks the prefix bucket only (score below 1, decreasing in K).",
        "* Two-server PIR leaks nothing to any single server (score exactly 1.0).",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    docs = ROOT / "docs"
    df = run_sweep()
    df.to_csv(docs / "results.csv", index=False)
    make_plots(df, docs / "plots")
    write_markdown(df, docs / "results.md", docs / "plots")
    print(f"wrote {docs / 'results.csv'} ({len(df)} rows)")
    print(f"wrote {docs / 'results.md'}")
    print(f"wrote {docs / 'plots'}/*.png")
    summary = (
        df.groupby("approach")[["privacy_score", "total_bytes", "latency_s"]]
        .mean()
        .round(4)
    )
    print("\nMean metrics per approach over the whole sweep:\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()