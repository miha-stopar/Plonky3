# p3-dft-metal

GPU-accelerated BabyBear NTT and fused DFT+Merkle for WHIR on Apple Silicon (Metal).

Ported from [whir-p3-metal](https://github.com/tcoratger/whir-p3-metal); see also the [ethresear.ch write-up](https://ethresear.ch/t/gpu-accelerated-whir-proving-on-apple-silicon/24762).

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
use p3_dft_metal::{GpuMmcs, MetalBabyBearDft, DftCommitFusion};
// WhirProver::commit_fused / open_fused / prove_fused
```

Example: `cargo run -p p3-whir --example whir_gpu --features gpu-metal --release`

## Shaders

`shaders/babybear_ntt.metal` — Montgomery field, radix DIF NTT, Poseidon2/Keccak Merkle, PoW.
