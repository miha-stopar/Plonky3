# WHIR GPU benchmarks (Apple Silicon)

Results below were measured on **Apple M1** with release builds using **thin LTO**, **`codegen-units = 1`**, and **`target-cpu=native`** (see workspace `Cargo.toml` and `.cargo/config.toml`).

Ported GPU code lives in [`p3-whir-metal`](../whir-metal/); see the [ethresear.ch write-up](https://ethresear.ch/t/gpu-accelerated-whir-proving-on-apple-silicon/24762) for architecture context (based on [whir-p3-metal](https://github.com/miha-stopar/whir-p3-metal)).

## What is being timed

| Bench | Metric | Fields | GPU modes |
|-------|--------|--------|-----------|
| [`whir_ethresearch_gpu`](whir_ethresearch_gpu.rs) | Full proof = **commit + open** | BabyBear, `PrefixProver` | Best of **gpu**, **gpu_fused**, **gpu_grind** |
| [`whir_grid_gpu`](whir_grid_gpu.rs) | Full proof | BabyBear + KoalaBear | **gpu_fused** only |
| [`whir_pcs_gpu`](whir_pcs_gpu.rs) | commit / prove / full separately | BabyBear + KoalaBear | **gpu_fused** only |

### GPU mode definitions

- **gpu** — `MetalBabyBearDft` + `GpuMmcs`, standard `commit` / `open` (GPU work per STIR round, not one fused buffer).
- **gpu_fused** — `commit_fused` / `open_fused` / `prove_fused` (transpose+NTT+Merkle fused in Metal where supported).
- **gpu_grind** — same as fused, but `GpuChallenger` races CPU and GPU on PoW grinding when difficulty ≥ 20 bits.

**Best GPU** = minimum median time among the three GPU modes (matches the ethresear.ch tables).

### Protocol knobs (grid / ethresearch benches)

- `security_level = 100`
- `pow_bits = 0` in `ProtocolParameters` (folding/query PoW still derived by `WhirConfig`)
- 3 runs per cell, **median** reported
- Parameter grid: `n ∈ {20, 22, 24}`, `fold` and `rate` as in the [post](https://ethresear.ch/t/gpu-accelerated-whir-proving-on-apple-silicon/24762) (29 configs at n=20/22, 5 at n=24)

## How to run

```bash
# Ethresearch-style BabyBear table (recommended for post comparison)
cargo build -p p3-whir --bench whir_ethresearch_gpu --features gpu-metal --release
./target/release/deps/whir_ethresearch_gpu-*

# BabyBear + KoalaBear grid (fused only)
cargo build -p p3-whir --bench whir_grid_gpu --features gpu-metal --release
WHIR_GRID_BABYBEAR_ONLY=1 ./target/release/deps/whir_grid_gpu-*   # BabyBear only
WHIR_GRID_KOALA_ONLY=1 ./target/release/deps/whir_grid_gpu-*      # KoalaBear only

# Criterion: commit / prove / full at fixed fold=4, rate=1
cargo bench -p p3-whir --bench whir_pcs_gpu --features gpu-metal -- babybear/full --noplot
```

---

## BabyBear — ethresear.ch-style (M1, Plonky3)

Layout: **PrefixProver**. Speedup = CPU median ÷ Best GPU median.

### n = 20

| fold | rate | CPU (ms) | Best GPU (ms) | Speedup | Best mode |
|------|------|----------|---------------|---------|-----------|
| 1 | 1 | 1156.6 | 300.1 | 3.85× | fused |
| 1 | 2 | 1930.6 | 337.5 | 5.72× | grind |
| 1 | 3 | 4232.2 | 559.8 | 7.56× | grind |
| 2 | 1 | 626.5 | 132.3 | 4.74× | grind |
| 2 | 2 | 1231.0 | 186.1 | 6.62× | grind |
| 2 | 3 | 2344.1 | 362.4 | 6.47× | fused |
| 4 | 1 | 646.0 | 395.7 | 1.63× | grind |
| 4 | 2 | 1349.5 | 381.3 | 3.54× | grind |
| 4 | 3 | 8987.7 | 1777.8 | 5.06× | grind |

### n = 22

| fold | rate | CPU (ms) | Best GPU (ms) | Speedup | Best mode |
|------|------|----------|---------------|---------|-----------|
| 1 | 1 | 4297.2 | 1138.9 | 3.77× | grind |
| 1 | 2 | 7542.5 | 1368.7 | 5.51× | grind |
| 1 | 3 | 14878.8 | 2171.4 | 6.85× | fused |
| 2 | 1 | 2605.6 | 565.7 | 4.61× | grind |
| 2 | 2 | 4989.5 | 823.8 | 6.06× | fused |
| 2 | 3 | 9730.1 | 1557.8 | 6.25× | grind |
| 3 | 1 | 1955.0 | 505.3 | 3.87× | grind |
| 3 | 2 | 5879.2 | 1222.3 | 4.81× | grind |
| 3 | 3 | 38367.1 | 4332.6 | 8.86× | grind |
| 4 | 1 | 3159.4 | 559.3 | 5.65× | grind |
| 4 | 2 | 9277.3 | 1398.9 | 6.63× | grind |
| 4 | 3 | 51664.8 | 5437.8 | 9.50× | grind |
| 6 | 1 | 3670.3 | 777.0 | 4.72× | grind |
| 6 | 2 | 92058.5 | 8800.3 | 10.46× | grind |
| 6 | 3 | 428955.2 | 41393.9 | 10.36× | grind |

### n = 24 (rate = 1 only)

| fold | rate | CPU (ms) | Best GPU (ms) | Speedup | Best mode |
|------|------|----------|---------------|---------|-----------|
| 1 | 1 | 17074.2 | 4667.2 | 3.66× | fused |
| 2 | 1 | 10435.9 | 2159.6 | 4.83× | fused |
| 3 | 1 | 10159.6 | 2542.0 | 4.00× | grind |
| 4 | 1 | 46770.0 | 5212.8 | 8.97× | grind |
| 6 | 1 | 52068.5 | 5569.4 | 9.35× | grind |

**Summary (29/29 configs):** speedup **1.63× – 10.46×**, mean **6.00×**.

Per-mode medians are in the bench log columns `gpu`, `fused`, `grind` (see [`whir_ethresearch_gpu.rs`](whir_ethresearch_gpu.rs) output).

---

## Comparison to ethresear.ch M1 reference

The post used `whir-p3-metal` + `bench.sh` (not this Plonky3 PCS harness). Absolute milliseconds differ; speedup ratios are closer at low fold.

| n | fold | rate | Post CPU | Post GPU | Post ↑ | Plonky3 CPU | Plonky3 GPU | Plonky3 ↑ |
|---|------|------|----------|----------|--------|-------------|-------------|-----------|
| 20 | 1 | 1 | 267 | 171 | 1.56× | 1157 | 300 | 3.85× |
| 20 | 2 | 1 | 127 | 75 | 1.69× | 627 | 132 | 4.74× |
| 22 | 1 | 1 | 1174 | 579 | 2.03× | 4297 | 1139 | 3.77× |
| 22 | 1 | 2 | 1938 | 1092 | 1.77× | 7543 | 1369 | 5.51× |
| 22 | 2 | 1 | 441 | 287 | 1.54× | 2606 | 566 | 4.61× |
| 22 | 4 | 1 | 166 | 128 | 1.30× | 3159 | 559 | 5.65× |
| 24 | 1 | 1 | 4153 | 2463 | 1.69× | 17074 | 4667 | 3.66× |
| 24 | 2 | 1 | 1814 | 1049 | 1.73× | 10436 | 2160 | 4.83× |

Post headline on M1: about **1.25× – 2.03×** with best-of-three GPU modes. Plonky3 on the same machine often shows **higher speedup ratios** because the CPU baseline in this harness is slower in absolute time, not because GPU times alone are dramatically lower than the post.

---

## KoalaBear — grid (M1, `gpu_fused` only)

Layout: **SuffixProver** (matches [`whir_pcs`](whir_pcs.rs)). Same `(n, fold, rate)` grid; `security_level = 100`.

### n = 20

| fold | rate | CPU (ms) | GPU (ms) | Speedup |
|------|------|----------|----------|---------|
| 1 | 1 | 1542.7 | 592.4 | 2.60× |
| 1 | 2 | 2436.2 | 495.7 | 4.91× |
| 1 | 3 | 4669.7 | 711.0 | 6.57× |
| 2 | 1 | 749.1 | 245.6 | 3.05× |
| 2 | 2 | 1312.6 | 312.7 | 4.20× |
| 2 | 3 | 2536.7 | 470.1 | 5.40× |
| 4 | 1 | 528.9 | 274.1 | 1.93× |
| 4 | 2 | 1642.2 | 2290.3 | 0.72× |
| 4 | 3 | 15617.9 | 14553.2 | 1.07× |

### n = 22

| fold | rate | CPU (ms) | GPU (ms) | Speedup |
|------|------|----------|----------|---------|
| 1 | 1 | 6325.9 | 2350.9 | 2.69× |
| 1 | 2 | 9943.2 | 2069.1 | 4.81× |
| 1 | 3 | 19257.2 | 2892.7 | 6.66× |
| 2 | 1 | 3070.0 | 945.3 | 3.25× |
| 2 | 2 | 5397.0 | 1141.4 | 4.73× |
| 2 | 3 | 10606.3 | 1964.4 | 5.40× |
| 3 | 1 | 2387.4 | 973.8 | 2.45× |
| 3 | 2 | 6948.2 | 3980.4 | 1.75× |
| 3 | 3 | 50433.9 | 46594.8 | 1.08× |
| 4 | 1 | 2413.4 | 1319.0 | 1.83× |
| 4 | 2 | 6593.7 | 7126.8 | 0.93× |
| 4 | 3 | 24949.0 | 30017.0 | 0.83× |
| 6 | 1 | 8380.2 | 6416.8 | 1.31× |
| 6 | 2 | 95204.8 | 37011.0 | 2.57× |
| 6 | 3 | — | — | failed (PoW witness) |

### n = 24 (rate = 1)

| fold | CPU (ms) | GPU (ms) | Speedup |
|------|----------|----------|---------|
| 1 | 27127.4 | 9595.3 | 2.83× |
| 2 | 12372.9 | 3769.3 | 3.28× |
| 3 | 11003.7 | 3519.6 | 3.13× |
| 4 | 50699.0 | 76384.3 | 0.66× |
| 6 | 37130.2 | 23514.0 | 1.58× |

**Summary (28/29 configs):** speedup **0.66× – 6.66×**, mean **2.89×** (fused path only).

At **n = 22, fold = 6, rate = 3** the prover failed with `failed to find proof-of-work witness`. Use `security_level ≤ 100` for KoalaBear on this grid; `128` can trip `DuplexChallenger::grind` field-order asserts at small `n`.

---

## BabyBear vs KoalaBear (same cell, fused only)

Fixed harness: `whir_pcs_gpu`, `FOLDING = 4`, `LOG_INV_RATE = 1`, `POW_BITS = 0`, Criterion medians on M1.

| n | | BabyBear CPU | BabyBear GPU | BB ↑ | Koala CPU | Koala GPU | KB ↑ |
|---|--|--------------|--------------|------|-----------|-----------|------|
| 14 | commit | 989 µs | 2.38 ms | 0.42× | 989 µs | 2.38 ms | 0.42× |
| 18 | commit | 16.3 ms | 6.90 ms | 2.4× | 13.1 ms | 3.13 ms | 4.2× |
| 20 | commit | 73.5 ms | 14.5 ms | 5.1× | 68.7 ms | 14.5 ms | 4.7× |
| 22 | commit | 283 ms | 57.5 ms | 4.9× | 280 ms | 57.5 ms | 4.9× |
| 22 | prove | 2960 ms | 2113 ms | 1.4× | 967 ms | 321 ms | 3.0× |
| 22 | full | — | — | — | — | — | see grid tables |

Koala **prove-only** can look faster than BabyBear at large `n` because the CPU suffix path is cheaper in that micro-bench, while **commit** speedups are similar. Compare **full proof** via `whir_ethresearch_gpu` / `whir_grid_gpu` for fair end-to-end numbers.

---

## Layout and fields (short)

- **PrefixProver** — `Witness::new_interleaved`; folds selector/outer structure first; used with **BabyBear** in GPU examples and ethresearch benches. See [`sumcheck/src/layout/prover/prefix.rs`](../../sumcheck/src/layout/prover/prefix.rs).
- **SuffixProver** — `Witness::new`; folds local/inner structure first; default in [`whir_pcs`](whir_pcs.rs) with **KoalaBear**. See [`sumcheck/src/layout/prover/suffix.rs`](../../sumcheck/src/layout/prover/suffix.rs).

---

## Caveats

1. **Not comparable ms-for-ms with the post** without running `whir-p3-metal`’s `bench.sh` on the same machine.
2. **Thermal / background load** — medians of 3 runs; long grids take hours.
3. **`gpu_grind`** only helps when derived PoW difficulty is large enough (see `GpuChallenger` in `whir-metal/src/gpu_dft.rs`).
4. Re-run after toolchain or protocol changes; pin commit hash when publishing numbers.
