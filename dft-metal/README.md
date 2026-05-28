# p3-dft-metal

GPU acceleration for WHIR on Apple Silicon (Metal): **NTT**, **Poseidon2/Keccak Merkle**, **fused commit pipelines**, and **PoW grinding** for BabyBear and KoalaBear.

> **Naming:** the crate and shader files (`*_ntt.metal`) are historical — they are not DFT-only. See [INTEGRATION.md](INTEGRATION.md) for architecture, code links, and WHIR wiring.

Ported from [whir-p3-metal](https://github.com/miha-stopar/whir-p3-metal); see also the [ethresear.ch write-up](https://ethresear.ch/t/gpu-accelerated-whir-proving-on-apple-silicon/24762).

## Features

- `MetalBabyBearDft` — `TwoAdicSubgroupDft<BabyBear>` with CPU fallback for small matrices
- `GpuMmcs` / `GpuKeccakMmcs` — Poseidon2 or Keccak Merkle on GPU
- `DftCommitFusion` — single Metal command buffer: transpose/pad → NTT → hash → compress
- `GpuChallenger` — GPU PoW grinding for high `pow_bits`

## Usage

Enable on macOS/iOS:

```toml
p3-whir = { path = "whir", features = ["gpu-metal"] }
```

```rust
use p3_dft_metal::{GpuMmcs, GpuKoalaMmcs, MetalBabyBearDft, MetalKoalaBearDft, DftCommitFusion};
// WhirProver::commit_fused / open_fused / prove_fused
```

**Fields:** BabyBear (`MetalBabyBearDft`, `GpuMmcs`) and KoalaBear (`MetalKoalaBearDft`, `GpuKoalaMmcs`).

Example: `cargo run -p p3-whir --example whir_gpu --features gpu-metal --release`

## Benchmarks

Full tables, reproduction commands, and comparison to the [ethresear.ch M1 results](https://ethresear.ch/t/gpu-accelerated-whir-proving-on-apple-silicon/24762) are in [`whir/benches/BENCHMARKS.md`](../whir/benches/BENCHMARKS.md).

**BabyBear on M1 (Plonky3, full proof, best of gpu / fused / grind):** 29-config grid, speedup about **1.6× – 10.5×** (mean **6.0×**). Example: `n = 22`, `fold = 1`, `rate = 1` → CPU **4297 ms**, Best GPU **1139 ms**, **3.77×**.

**KoalaBear on M1 (`gpu_fused` only, SuffixProver):** same grid, **28/29** configs OK, speedup about **0.7× – 6.7×** (mean **2.9×**); one cell fails at `n = 22`, `fold = 6`, `rate = 3`.

```bash
cargo build -p p3-whir --bench whir_ethresearch_gpu --features gpu-metal --release
./target/release/deps/whir_ethresearch_gpu-*
```

## Documentation

- [INTEGRATION.md](INTEGRATION.md) — how `p3-whir` / `p3-sumcheck` call into this crate; links to NTT, Merkle, grind, and fused paths
- [../whir/benches/BENCHMARKS.md](../whir/benches/BENCHMARKS.md) — M1 benchmark tables and reproduction

## Shaders

One `.metal` file per field (all kernels for that field live in one translation unit):

- `shaders/babybear_ntt.metal` — field arithmetic, NTT, transpose/pad, Poseidon2 Merkle, Keccak Merkle, PoW grind
- `shaders/koalabear_ntt.metal` — KoalaBear equivalents (different modulus + Poseidon internal layer)
