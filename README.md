# Comparative Study of Privacy-Preserving Information Retrieval

This repository compares three approaches for retrieving records from a
server-side dataset while studying the **privacy**, **communication** and
**computation** trade-offs that each one exposes:

| # | Approach | Module | Leaked identity bits | Example |
|---|----------|--------|----------------------|---------|
| 1 | **Direct Access** (baseline) | `src/direct` | `log2(K)` (everything) | `SELECT * FROM msgs WHERE id = ?` |
| 2 | **K-Anonymity** (prefix-based query hiding) | `src/kanon` | `log2(k)` (bucket size) | send only a prefix, server returns a bucket of `k` rows |
| 3 | **Two-Server PIR** (private information retrieval) | `src/pir` | `0` (unconditional) | XOR of sub-sampled messages from two non-colluding servers |

All three approaches are benchmarked against the same dataset through a
single comparison runner and a Streamlit dashboard.

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt` (`numpy`, `pytest`, `pytest-cov`,
  `streamlit`, `pyyaml`, `matplotlib`, `pandas`)

## Setup

```bash
cd pir-comparative-study
python -m pip install -r requirements.txt
```

## Quick start

Verify the installation:

```bash
python -c "import src"          # must succeed
pytest tests/ -q                # run the whole test suite
```

Run the step-by-step demo (K=8, N=2, M=2, L=16):

```bash
python scripts/run_demo.py
```

Sweep K, N and M and regenerate `docs/results.md`, `docs/results.csv`
and the experiment plots:

```bash
python scripts/run_experiments.py
```

Launch the comparison dashboard:

```bash
streamlit run src/comparison/dashboard.py
```

## Project layout

```
pir-comparative-study/
├── config.yaml            # all run parameters (N, K, M, L, q, seed, ...)
├── src/
│   ├── common/            # Message, dataset loaders, metrics
│   ├── direct/            # baseline: no privacy
│   ├── kanon/             # prefix-based k-anonymity
│   ├── pir/               # two-server XOR PIR
│   └── comparison/        # runner + Streamlit dashboard
├── tests/                 # pytest suite (one file per module/phase)
├── scripts/               # demo and experiment drivers
└── docs/
    ├── design.md          # architecture, algorithms, pseudocode
    └── results.md         # experiment tables and plots
```

## Core ideas

- **Messages** are `(id, payload)` pairs with `payload` a NumPy array of
  integers in `[0, q)`. For `q == 2` the payload is a bit-vector and all
  private-retrieval algebra is vectorized XOR (`numpy.bitwise_xor`).
- **Direct access** reveals the exact `msg_id` to the server (privacy = 0).
- **K-Anonymity** hides the query inside a bucket of `k >= 1` messages that
  share a common prefix; the server learns only the prefix.
- **Two-server PIR** (Chor et al. XOR scheme) sends one random query vector
  to each of two non-colluding servers so that neither one can distinguish
  the requested message from any other; the client XORs the two answers to
  recover the exact message (bit-exact recovery).

## Testing

```bash
pytest tests/              # all phases
pytest --cov=src tests/    # coverage report (target > 80%)
```

## Documentation

- [`docs/design.md`](docs/design.md) — architecture, module descriptions and
  algorithm pseudocode.
- [`docs/results.md`](docs/results.md) — regenerated tables and plots from
  `scripts/run_experiments.py`.