# PIR-PSI Design — Achieving the Capacity of PIR with Private Side Information

This document describes the **PIR with Private Side Information (PIR-PSI)**
scheme implemented in `src/pir_psi/`, following:

> L. Erhili and A. Heidarzadeh, *"Achieving Capacity of PIR with Private Side
> Information with Low Sub-packetization and without MDS Codes"*,
> ISIT 2024 / arXiv:2401.16630.

The module is a sibling of the classical two-server Chor PIR in `src/pir`,
which remains the `M = 0` baseline.

---

## 1. Problem statement

- `N >= 2` **non-colluding** servers store identical copies of `K` messages
  `X_1, ..., X_K`, each an `L`-symbol vector over `F_q`.
- A client **privately holds** `M` messages `X_S = {X_i : i in S}` as side
  information and wants to retrieve one more message `X_W`, `W not in S`.
- The servers are initially unaware of the identities in `S`.

The scheme must satisfy two conditions:

1. **Privacy:** for every server `n`, `I(W, S; Q_n) = 0`.  *Both* the demand
   and the side-information set must be hidden.
2. **Recoverability:** `H(X_W | A_[N], Q_[N], X_S) = 0`.

The **rate** is `L / sum_n H(A_n | Q_n)`, and the **capacity** is the maximum
achievable rate.

---

## 2. Main result

> **Theorem (Erhili & Heidarzadeh, 2024).**  The capacity of PIR-PSI is
>
> ```
> R = (N^{K-M} - N^{K-M-1}) / (N^{K-M} - 1),
> ```
>
> whenever `N >= M + 1` and `(N-1) | L`.

The scheme attaining it uses:

- **sub-packetization `N-1`**, independent of `K`;
- **only XOR operations** (no MDS codes, no field extension), valid for any
  field including `F_2`;
- one coded sub-packet downloaded per server; each sub-packet of the demand
  is recovered from a **distinct** server.

### 2.1 Sanity checks

| `(N, K, M)` | `R` | Note |
|-------------|-----|------|
| `(4, 5, 2)` | `16/21` | worked example of the paper |
| `(2, 4, 0)` | `8/15` | finite-length PIR (Sun–Jafar); `K -> inf` gives Chor `1/2` |
| `(2, inf, 0)` | `1/2` | classical Chor two-server XOR PIR |

---

## 3. Message sub-packeting

Each message `X_i` is split into `N-1` **contiguous sub-packets** of `L/(N-1)`
symbols:

```
X_i = (X_{i,1}, X_{i,2}, ..., X_{i,N-1}).
```

Sub-packet index `0` is reserved for the **degenerate all-zero** block
`X_{i,0} := 0`.  A query is a vector `v in [0:N-1]^K`; server `n` answers

```
Y_n = sum_{i=1}^{K} X_{i, v_n(i)}      (X_{i,0} treated as zero),
```

and stays silent (`Y_n = 0`) when `v_n` is all-zero.

---

## 4. Non-uniform query distribution (Lemma 1)

The distribution is defined over `(I, J) in [0:M] x [0:K-M-1]`, where

- `I` = number of side-information sub-packets mixed into the query,
- `J` = number of interference sub-packets mixed into the query.

With `m_{0,0} := 1`, `m_{i,j} := 0` for `1 <= i+j <= M`, and for
`M+1 <= i+j <= K-1`:

```
m_{i,j} = sum_{k=0}^{i+j-M-1} (-1)^k * C(M+k-1, k) * (N-1)^{i+j-M-k},
```

the probability of each pair is

```
P_{i,j} = C(M, i) * C(K-M-1, j) * m_{i,j} / N^{K-M-1}.
```

**Lemma 1** proves `P_{i,j} >= 0` and `sum P_{i,j} = 1`.  The distribution is
*non-uniform*: it depends only on the **support size** `i+j` of the resulting
query, which is exactly what makes every server's marginal view independent of
`(W, S)`.

> Verification in code: `probability_matrix(M, K, N)` asserts the sum is `1`;
> `combinatorial_m(i, j, M, N)` implements `m_{i,j}` with the generalized
> binomial convention `C(-1, 0) = 1` needed for the `M = 0` reduction.

---

## 5. Query construction (Algorithm 1)

Given `W`, `S`, and randomness:

1. Draw a random permutation `pi` of `{1, ..., N-1}` and build `N-1` vectors
   `a_1, ..., a_{N-1}` with `a_m(W) = pi(m)` and `a_m(i) = 0` otherwise.
   These carry the `N-1` demand sub-packets.
2. Draw `b` with `supp(b) = S` (each `b(i) in {1, ..., N-1}` uniformly).
3. Sample `(I, J) ~ P`.  Pick a random `I`-subset `R` of `S` and set
   `b_0(i) = b(i)` for `i in R`, else `0`.  For `I > 0`, form `I` vectors
   `b_1, ..., b_I` with supports `R_m = R \ {r_m}` (the `I` single-element
   deletions of `R`), in random order.
4. Draw interference subset `T` (size `J`) of `[K] \ ({W} u S)` and a vector
   `c` with `supp(c) = T`.
5. Draw `theta in {0,1}` with `P(theta = 0) = (M-I)/(M-I+1)`.
6. Assemble `u_1, ..., u_N`:
   ```
   u_1 = b_0 + c
   theta = 0 :  u_{m+1} = a_m + b   + c   for m in [N-1]
   theta = 1 :  u_{m+1} = a_m + b_m + c   for m in [I]
                u_{m+1} = a_m + b   + c   for m in [I+1 : N-1]
   ```
   All supports are disjoint, so the sums are unambiguous set-unions.
7. Draw a random permutation `sigma` of `[N]` and send `v_n = u_{sigma(n)}` to
   server `n`.

The client keeps `(pi, sigma, theta, I, J, R, {r_m}, b, c)` secret for decoding.

---

## 6. Decoding (Lemma 3)

Let `Z_j = Y_{sigma^{-1}(j)}` be the answers reordered back to `u_j`.  Since
`u_1 = b_0 + c` contains **no** demand sub-packet and
`u_{m+1}` contains demand sub-packet `pi(m)`, and since the side terms `b_0`
and `b_m`/`b` are known to the client (it holds `X_S` and chose the
coordinates), it cancels them exactly:

```
X_{W, pi(m)} = Z_{m+1} - Z_1 - side_m + side_0,     (all sums are XOR)
```

where

```
side_0 = sum_{i in R}        X_{i, b(i)}
side_m = sum_{i in R_m}      X_{i, b(i)}   if theta = 1 and m <= I
       = sum_{i in S}        X_{i, b(i)}   otherwise.
```

Reassembling the `N-1` recovered sub-packets gives `X_W` bit-exactly

> `src/pir_psi/decoder.py::decode` implements this equation directly.

### 6.1 Privacy argument (Lemma 2)

For a fixed query vector `v` of support size `s`, one shows
`P(Q_n = v | W, S)` depends only on `s` (and not on `W` or `S`).  The four
cases `s = 0`, `s = M+1`, `M+1 < s < K`, `s = K` all collapse to the same
expression.  Hence `I(W, S; Q_n) = 0` for every server.

---

## 7. Rate and achievability

Each non-zero query produces an answer of `L/(N-1)` symbols, and

```
P(v_n ≠ 0) = 1 - P_{0,0}/N = 1 - 1/N^{K-M}.
```

Therefore the expected download is
`N (1 - 1/N^{K-M}) L/(N-1)`, giving

```
R = L / download = (N-1) / (N - N / N^{K-M}) = (N^{K-M} - N^{K-M-1}) / (N^{K-M} - 1).
```

The scheme is achievable whenever `N >= M + 1` and `(N-1) | L`
(`src/pir_psi/capacity.py::is_achievable`).

---

## 8. Module map

| File | Responsibility |
|------|----------------|
| `src/pir_psi/query.py` | `combinatorial_m`, `probability_matrix`, `sample_query_parameters`, `build_query` (Algorithm 1), `permute_vectors`, `generate_query_vectors` |
| `src/pir_psi/server.py` | `PIRPSIServer` — sub-packet XOR answers, `None` for all-zero queries |
| `src/pir_psi/decoder.py` | `decode` — side-information cancellation |
| `src/pir_psi/client.py` | `PIRPSIClient` — query generation + retrieval + stats |
| `src/pir_psi/capacity.py` | `capacity_rate`, `is_achievable`, `expected_download_bits`, `measure_actual_rate` |

---

## 9. Comparison: Chor PIR vs. Erhili PIR-PSI vs. Chen 2020

| Property | Chor PIR (`src/pir`) | **Erhili PIR-PSI** (`src/pir_psi`) | Chen et al. 2020 |
|----------|----------------------|-------------------------------------|------------------|
| Servers | `N = 2` (any `N`) | `N >= M+1` | `N >= M+1` |
| Side information | non-private (`M = 0` baseline) | **private**, `M >= 0` | private, `M >= 1` |
| Rate | `1/2` | `(N^{K-M}-N^{K-M-1})/(N^{K-M}-1)` (capacity) | capacity |
| Sub-packetization | `2` | **`N-1`** (constant in `K`) | `N^K` (exponential) |
| Field | `F_2` | **any `F_q`, incl. `F_2`** | large `q` (exponential) |
| MDS code | no | **no** | yes (split length `(2N^K-N^M-1)/(N-1)`) |
| Ops | XOR | XOR | field mult. + MDS |
| Divisibility | `2 | L` | `(N-1) | L` | `N^K | L` |

PIR-PSI strictly improves the rate over Chor PIR whenever `M > 0` while
keeping the same XOR-only, low-complexity flavour.

---

## 10. Open regimes

The converse bound `R` is **not tight in general**, and the Erhili scheme does
not apply, when:

- `N <= M` (more side-information messages than servers): capacity is open;
- `(N-1) ∤ L`: capacity is open — the message cannot be split into `N-1`
  equal sub-packets;
- `q = 2` with `L = 1` (no room for sub-packeting); for example the paper
  notes `(N=2, M=2, L=1, q=2)` has capacity `1` at `K=3` but `5/8 < 2/3` at
  `K=4`.

In the implemented scheme, the runner automatically selects the smallest
`N >= M+1` with `(N-1) | L` when the configured `N` is not achievable
(`src/comparison/runner.py::_choose_pir_psi_N`).
