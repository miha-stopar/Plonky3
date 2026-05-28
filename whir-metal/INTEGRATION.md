# GPU WHIR integration (Plonky3)

How Metal acceleration connects to WHIR proving in crate **`p3-whir-metal`**. Each field has one Metal source file (`shaders/*_whir.metal`) with **NTT, Poseidon2, Keccak, transpose, and PoW** kernels — see [Shader layout](#shader-layout).

Upstream reference: [whir-p3-metal](https://github.com/miha-stopar/whir-p3-metal), [ethresear.ch write-up](https://ethresear.ch/t/gpu-accelerated-whir-proving-on-apple-silicon/24762).

## Feature wiring

```text
p3-whir/gpu-metal
    ├── p3-whir-metal/gpu-metal     (Metal kernels + GpuMmcs + GpuChallenger)
    └── p3-sumcheck/gpu-metal      (commit_base_fused → DftCommitFusion)
```

Enable:

```toml
p3-whir = { features = ["gpu-metal"] }
```

## Public surface (`p3-whir-metal`)

| Type | Role |
|------|------|
| [`MetalBabyBearDft`](src/gpu_dft.rs) | `TwoAdicSubgroupDft<BabyBear>` — GPU NTT with CPU fallback |
| [`MetalKoalaBearDft`](src/gpu_koala_dft.rs) | Same for KoalaBear |
| [`GpuMmcs`](src/gpu_dft.rs) | `Mmcs` + [`DftCommitFusion`](src/gpu_dft.rs) — Poseidon2 Merkle on GPU |
| [`GpuKoalaMmcs`](src/gpu_koala_dft.rs) | KoalaBear MMCS |
| [`GpuKeccakMmcs`](src/gpu_dft.rs) / [`GpuKoalaKeccakMmcs`](src/gpu_koala_dft.rs) | Keccak Merkle variant |
| [`GpuChallenger`](src/gpu_dft.rs) | `DuplexChallenger` + GPU PoW (`poseidon2_pow_grind`) |

Exports: [`src/lib.rs`](src/lib.rs).

## Where optimizations live

### NTT / DFT

| Layer | Link |
|-------|------|
| MSL kernels | [`shaders/babybear_whir.metal`](shaders/babybear_whir.metal) — `bb_dif_r*`, `bb_ntt_*`, `bb_ntt_stockham`, bit-reversal, shared-memory butterflies |
| Koala MSL | [`shaders/koalabear_whir.metal`](shaders/koalabear_whir.metal) — `kb_*` equivalents |
| Rust dispatch | [`MetalBabyBearDft`](src/gpu_dft.rs) (`impl TwoAdicSubgroupDft`), pipeline table in `MetalInner` |
| Fused encode | [`transpose_pad_dft_and_commit`](src/gpu_dft.rs) on [`DftCommitFusion`](src/gpu_dft.rs) |

### Merkle (Poseidon2)

| Layer | Link |
|-------|------|
| MSL | `poseidon2_hash_leaves`, `poseidon2_merkle_compress`, `poseidon2_hash_and_compress`, `poseidon2_hash4_compress3`, `poseidon2_merkle_simd` in [`babybear_whir.metal`](shaders/babybear_whir.metal) |
| Rust | [`GpuMmcs`](src/gpu_dft.rs) — `commit_matrix` / fused paths; [`build_poseidon2_constants`](src/gpu_dft.rs) |
| GPU-backed tree | [`merkle-tree/src/merkle_tree.rs`](../merkle-tree/src/merkle_tree.rs) — `from_parts_gpu_backed`, `from_parts_gpu_backed_with_leaves` |

### Merkle (Keccak)

| Layer | Link |
|-------|------|
| MSL | `keccak_hash_*`, `keccak_merkle_compress*` in [`babybear_whir.metal`](shaders/babybear_whir.metal) |
| Rust | [`GpuKeccakMmcs`](src/gpu_dft.rs) |

### Transpose / pad (prefix layout)

| Layer | Link |
|-------|------|
| MSL | `bb_transpose_pad`, `bb_transpose_pad_packed` |
| Rust | [`DftCommitFusion::transpose_pad_dft_and_commit`](src/gpu_dft.rs), used from [`sumcheck/src/commit.rs`](../sumcheck/src/commit.rs) `commit_base_fused` |

### PoW grinding

| Layer | Link |
|-------|------|
| MSL | `poseidon2_pow_grind` in [`babybear_whir.metal`](shaders/babybear_whir.metal) |
| Rust | [`GpuChallenger::grind`](src/gpu_dft.rs) — races CPU rayon grind vs GPU; used when bench mode is `gpu_grind` |

### Constraint combine (optional GPU)

| Layer | Link |
|-------|------|
| MSL | `bb_combine_select` / `kb_combine_select` |
| Rust | [`gpu_combine_select`](src/gpu_dft.rs) on `MetalBabyBearDft` |

## WHIR integration paths

Three bench modes map to these code paths (see [`whir/benches/BENCHMARKS.md`](../whir/benches/BENCHMARKS.md)):

### 1. Standard GPU (`commit` / `open` / `prove`)

```text
WhirProver::commit  →  Layout::commit  →  commit_base (CPU layout, GPU dft/mmcs per step)
WhirProver::open    →  prove           →  commit_extension each STIR round (CPU path)
```

- Adapter: [`whir/src/pcs/adapter.rs`](../whir/src/pcs/adapter.rs) (`MultilinearPcs::commit` / `open`)
- Prover: [`whir/src/pcs/prover/mod.rs`](../whir/src/pcs/prover/mod.rs) (`prove`, `round`)

GPU runs NTT and Merkle **per step** with sync between DFT and hash.

### 2. Fused GPU (`commit_fused` / `open_fused` / `prove_fused`) — recommended

```text
WhirProver::commit_fused
    →  Layout::commit_fused
    →  commit_base_fused (sumcheck)
           →  transpose_pad_dft_and_commit  OR  dft_and_commit
                →  single Metal command buffer (NTT + Poseidon Merkle)

WhirProver::prove_fused
    →  prove_fused
    →  each STIR round: commit_extension_fused (whir)
           →  GpuMmcs::transpose_pad_dft_algebra_and_commit (extension field)
```

Key call sites:

| Step | File | Symbol |
|------|------|--------|
| PCS commit (fused) | [`whir/src/pcs/adapter.rs`](../whir/src/pcs/adapter.rs) | `commit_fused`, `open_fused` |
| PCS prove (fused) | [`whir/src/pcs/prover/mod.rs`](../whir/src/pcs/prover/mod.rs) | `prove_fused`, `round_fused` |
| Initial commit | [`sumcheck/src/commit.rs`](../sumcheck/src/commit.rs) | `commit_base_fused` |
| Layout hook | [`sumcheck/src/layout/prover/prefix.rs`](../sumcheck/src/layout/prover/prefix.rs), [`suffix.rs`](../sumcheck/src/layout/prover/suffix.rs) | `commit_fused` |
| STIR extension commit | [`whir/src/pcs/committer/writer.rs`](../whir/src/pcs/committer/writer.rs) | `commit_extension_fused` |

### 3. Fused + GPU grind (`GpuChallenger`)

Same as fused, but the prover uses [`GpuChallenger`](../whir-metal/src/gpu_dft.rs) instead of `DuplexChallenger` so Fiat–Shamir PoW can use the GPU kernel when difficulty is high enough.

Bench: [`whir/benches/whir_ethresearch_gpu.rs`](../whir/benches/whir_ethresearch_gpu.rs).

## What stays on CPU

- **Sumcheck rounds** — [`sumcheck/src/strategy.rs`](../sumcheck/src/strategy.rs), layout provers after `into_sumcheck`
- **WHIR protocol logic** — OOD points, query opening, verifier
- **Fiat–Shamir** observe/sample (except `GpuChallenger::grind`)
- **Small matrices** — GPU path may decline; CPU fallback in `MetalBabyBearDft`

## Shader layout

There is **one Metal source file per field**, not one file per algorithm:

| File | Contents |
|------|----------|
| [`shaders/babybear_whir.metal`](shaders/babybear_whir.metal) | Montgomery field ops, **all NTT kernels**, transpose/pad, **Poseidon2 Merkle**, **Keccak Merkle**, **PoW grind**, combine_select |
| [`shaders/koalabear_whir.metal`](shaders/koalabear_whir.metal) | KoalaBear equivalents (`kb_*`, Koala Poseidon internal layer) |

Loaded via `include_str!` in [`gpu_dft.rs`](src/gpu_dft.rs) / [`gpu_koala_dft.rs`](src/gpu_koala_dft.rs).

## Examples and benches

| Artifact | Purpose |
|----------|---------|
| [`whir/examples/whir_gpu.rs`](../whir/examples/whir_gpu.rs) | Smoke test: fused commit |
| [`whir/benches/whir_pcs_gpu.rs`](../whir/benches/whir_pcs_gpu.rs) | Criterion: commit / prove / full, BabyBear + KoalaBear |
| [`whir/benches/whir_ethresearch_gpu.rs`](../whir/benches/whir_ethresearch_gpu.rs) | Post-style grid, best-of-3 GPU modes |
| [`whir/benches/whir_grid_gpu.rs`](../whir/benches/whir_grid_gpu.rs) | Full grid, fused only |
| [`whir/benches/BENCHMARKS.md`](../whir/benches/BENCHMARKS.md) | Published numbers |
