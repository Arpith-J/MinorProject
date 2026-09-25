# Design Document

A comparative study of three approaches to privacy-preserving information
retrieval.  This document describes the architecture, the privacy model, and
the algorithm of every approach, together with the exact formulas used by
the metrics.

## 1. Overview

We study the following protocol: a **client** wants to retrieve one record
from a server-held dataset without (fully) revealing which record it wants.
Three strategies are compared:

| Approach       | What the server learns              | Privacy score |
|----------------|--------------------------------------|---------------|
| Direct access  | the exact `msg_id`                   | 0             |
| K-anonymity    | a prefix shared by `k` messages      | `1 - log2(k)/log2(K)` |
| Two-server PIR | nothing (information-theoretically)  | 1             |

All approaches retrieve the **same bit vector** (bit-exact recovery is
asserted by the test suite), so a fair comparison of privacy, communication,
computation and latency is possible.

## 2. Repository layout

```
src/
├── common/        # Message model, dataset loaders, metrics
├── direct/        # baseline: point lookup, zero privacy
├── kanon/         # prefix bucket k-anonymity
├── pir/           # two-server XOR private information retrieval
└── comparison/    # runner (uniform metrics) + Streamlit dashboard
scripts/           # run_demo.py, run_experiments.py
tests/             # one pytest file per module / development phase
docs/              # design.md (this file), results.md (generated)
```

`config.yaml` is the single source of protocol parameters; every function
receives the configuration explicitly (no global state).

## 3. Common model (`src/common`)

### 3.1 `Message`

- `msg_id: int`, `payload: np.ndarray` of `np.uint8`, alphabet `[0, q)`,
  `length`.

- `Message.split_subpackets(n)` splits the payload into `n` equal contiguous
  chunks (requires `length % n == 0`), returning `n` messages carrying the
  same id.  With the canonical two-server PIR the sub-packetization factor is
  `N - 1 = 1` (no split), which is why the XOR answers are full-length.

- `Message.to_binary()` returns the payload for `q == 2` (bit-vectors).
  All private-retrieval algebra is vectorized XOR (`numpy.bitwise_xor`).

- `to_bytes()` / `from_bytes()` serialize the payload and are used to measure
  the real communication cost.

### 3.2 Dataset

- `load_synthetic(K, L, q, seed)` — reproducible random dataset.
- `load_csv(path)` — CSV with header `id,payload` (`payload` = comma
  separated field values).
- `get_ground_truth(msg_id, dataset)` — reference lookup used to assert
  bit-exact recovery.

### 3.3 Metrics (`metrics.py`)

Let `K` be the number of messages and `k` the anonymizing bucket size.

- `privacy_score(leakage, identity) = max(0, 1 - leakage / identity)`
  where `identity = log2(K)` is the entropy of choosing one of `K` ids.
  - Direct: `leakage = log2(K)` ⇒ score `0`.
  - K-anonymity: `leakage = log2(k)` ⇒ score `1 - log2(k)/log2(K)`.
  - PIR: `leakage = 0` ⇒ score `1`.

- `communication_cost(sent, received)` → `{bytes_sent, bytes_received,
  total_bytes}` using real serialized sizes.
- `computation_cost(num_ops, op_type)` → operation counts per type.
- `latency(start, end)` → wall-clock seconds (min over repeated runs, see
  `config["latency_repeats"]`).

## 4. Approach 1 — Direct access (`src/direct`)

**Protocol.**  Client sends `msg_id`; server returns `messages[msg_id]` and
logs the id.

**Security.**  None: the server can determine the exact record.  Leaked bits
= `log2(K)`, privacy score `0`.  This is the baseline every other approach
is measured against.

## 5. Approach 2 — Prefix-based k-anonymity (`src/kanon`)

**Idea.**  Widening the query: instead of sending the exact id, the client
sends a *prefix* that is shared by `k - 1` other ids.  The server answers
with the whole bucket; the client filters locally.

**Bucket construction** (`bucket.py`).

```
width   = ceil(log_base(K))
prefix  = leading(prefix_len, digits(msg_id, base, width))
bucket  = { msg_id : prefix }
```

For `K = 16`, `base = 2`, `prefix_len = 2` the buckets `"00", "01", "10",
"11"` each hold 4 ids.

**Protocol.**

```
client: prefix = compute_prefix(msg_id, prefix_len, base, width)
client → server: prefix
server → client: list of messages whose ids start with prefix
client: pick the message with msg_id == target
```

**Security.**  The server learns only the prefix, i.e. the target is one of
`k` candidates ⇒ leaked bits `log2(k)`, privacy score
`1 - log2(k)/log2(K)`.  Increasing `prefix_len` shrinks buckets, which
increases anonymity *k* but reduces the number of candidates the server
must consider — a direct tunable trade-off.

## 6. Approach 3 — Two-server PIR (`src/pir`)

### 6.1 Scheme (Chor et al., 1995, XOR variant)

Two **non-colluding** servers store identical copies of all `K` messages
over `GF(2)`.  To retrieve message `d`:

```
d        := target id
S        := random subset of [K] \ (side_info ∪ {d})
q0[m]    := 1 if m in S else 0                    (server 0 query)
q1       := q0 XOR e_d                             (server 1 query)
a0       := XOR of all messages selected by q0     (server 0 answer)
a1       := XOR of all messages selected by q1     (server 1 answer)
recover  := a0 XOR a1
```

Correctness: the symmetric difference of the two selected sets is exactly
`{d}`, so every other message is XOR-ed in an even number of times and
cancels:

```
a0 XOR a1 = (XOR_{m in S} W_m) XOR (XOR_{m in S△{d}} W_m) = W_d
```

Privacy: `q0` is a uniformly random vector, hence independent of `d`;
`q1 = q0 XOR e_d` is also uniformly random.  A single server therefore gets
zero information about `d` (information-theoretic privacy; per-server
5 download = one full-length answer, giving the 1/2 rate).

The all-zero vector appears with probability `2^-K`; the corresponding
answer is `None` (see `PIRServer.answer_zero()`), and the client treats a
missing answer as the zero block — preserving exact uniformity (forcing
always-non-zero vectors would leak `e_d`'s position to server 1).

### 6.2 Side information

The client declares a set `S₀` (size `M`) of messages it already holds as
*non-private* side information: those coordinates are forced to `0` in both
queries (they cancel in the XOR anyway).  The servers learn `S₀` but remain
agnostic about `d in [K] \ S₀`.  This reduces the query support size and the
number of XOR operands (computation) while keeping recovery exact.  A
description of the `M`-sensitive communication path is in `results.md`.

### 6.3 More than two servers

Extra servers receive independent uniform query vectors (each server's view
stays independent of `d`), but only answers 0 and 1 are needed for decoding.
This gives a correct, private scheme for any `N >= 2`.

## 7. Comparison runner (`src/comparison/runner.py`)

`run_all(msg_id, side_info_ids, config)`:

1. prepares the dataset once (`prepare_dataset`);
2. runs the three approaches against the same messages;
3. collects, per approach:
   - `privacy` → `{leakage_bits, privacy_score}`,
   - `communication` → `{bytes_sent, bytes_received, total_bytes}`,
   - `computation` → list of `{op_type, num_ops}`,
   - `latency_s` → min over `latency_repeats` runs,
   - `recovered` → bit-exactness check against the ground truth.

`summarize` flattens this into a DataFrame consumed by the Streamlit
dashboard and the experiment scripts.

## 8. Streamlit dashboard (`src/comparison/dashboard.py`)

Sidebar controls for `N, K, M, L, q, prefix_len, seed, latency_repeats`,
a target `msg_id`, and a per-approach or "All" run button.  Results are shown
as a table plus three matplotlib panels (privacy, latency, communication).
The pure helpers (`build_report`, `plot_metrics`, `run_comparison`,
`_build_config`) are Streamlit-free and unit-tested; `main()` runs all
widget code and is only exercised via `streamlit run`.

## 9. Reproducibility

All randomness flows through a configured `random.Random(seed)`; the PIR
client and the dataset generator share the seed from `config.yaml`, so every
run and every experiment row is reproducible.