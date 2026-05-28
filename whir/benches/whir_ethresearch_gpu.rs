//! BabyBear WHIR grid benchmark in ethresear.ch table format.
//!
//! Full proof (commit + open), 3-run medians, CPU vs best of:
//! - `gpu`      — Metal DFT + GpuMmcs, standard commit/open
//! - `gpu_fused`— commit_fused / open_fused
//! - `gpu_grind`— fused path + [`GpuChallenger`] (GPU PoW when bits >= 20)
//!
//! Build (thin LTO + native CPU flags via workspace `.cargo/config.toml`):
//!   `cargo build -p p3-whir --bench whir_ethresearch_gpu --features gpu-metal --release`
//!   `./target/release/deps/whir_ethresearch_gpu-*`
//!
//! Published results: see `BENCHMARKS.md` in this directory.

use std::time::{Duration, Instant};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_dft_metal::{GpuChallenger, GpuMmcs, MetalBabyBearDft};
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
};
use p3_whir::pcs::prover::WhirProver;
use p3_whir::sumcheck::layout::{Layout, PrefixProver, Table};
use p3_whir::sumcheck::{OpeningProtocol, TableShape, TableSpec};
use rand::SeedableRng;
use rand::rngs::SmallRng;

const NUM_EVALUATIONS: usize = 1;
const POW_BITS: usize = 0;
const SECURITY_LEVEL: usize = 100;
const RUNS: usize = 3;

#[derive(Clone, Copy)]
struct GridConfig {
    n: usize,
    fold: usize,
    rate: usize,
}

fn grid_configs() -> Vec<GridConfig> {
    let mut out = Vec::new();
    for &n in &[20usize, 22, 24] {
        let folds: &[usize] = if n == 20 { &[1, 2, 4] } else { &[1, 2, 3, 4, 6] };
        let rates: &[usize] = if n == 24 { &[1] } else { &[1, 2, 3] };
        for &fold in folds {
            for &rate in rates {
                out.push(GridConfig { n, fold, rate });
            }
        }
    }
    out
}

fn round_log_inv_rates(
    num_variables: usize,
    folding_factor: &FoldingFactor,
    starting_rate: usize,
) -> Vec<usize> {
    let (num_rounds, _) = folding_factor.compute_number_of_rounds(num_variables);
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = starting_rate;
    for round in 0..num_rounds {
        rate += folding_factor.at_round(round) - 1;
        rates.push(rate);
    }
    rates
}

fn median_ms(samples: &mut [Duration]) -> f64 {
    samples.sort();
    samples[RUNS / 2].as_secs_f64() * 1000.0
}

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MerkleHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MerkleCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type Duplex = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type CpuMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type CpuDft = Radix2DFTSmallBatch<F>;
type GpuMmcsType = GpuMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type WhirLayout = PrefixProver<F, EF>;

type CpuPcs = WhirProver<EF, F, CpuDft, CpuMmcs, Duplex, WhirLayout>;
type GpuPcs = WhirProver<EF, F, MetalBabyBearDft, GpuMmcsType, Duplex, WhirLayout>;
type GpuGrindPcs = WhirProver<EF, F, MetalBabyBearDft, GpuMmcsType, GpuChallenger, WhirLayout>;

struct Fixture {
    cpu: CpuPcs,
    gpu: GpuPcs,
    gpu_grind: GpuGrindPcs,
    witness: <CpuPcs as MultilinearPcs<EF, Duplex>>::Witness,
    protocol: OpeningProtocol,
    domain_separator: DomainSeparator<EF, F>,
    perm: Perm,
    gpu_dft: MetalBabyBearDft,
}

impl Fixture {
    fn new(cfg: GridConfig) -> Self {
        let mut perm_rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut perm_rng);
        let hash = MerkleHash::new(perm.clone());
        let compress = MerkleCompress::new(perm.clone());
        let folding_factor = FoldingFactor::Constant(cfg.fold);
        let params = ProtocolParameters {
            security_level: SECURITY_LEVEL,
            pow_bits: POW_BITS,
            round_log_inv_rates: round_log_inv_rates(cfg.n, &folding_factor, cfg.rate),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: cfg.rate,
        };
        let config = WhirConfig::<EF, F, Duplex>::new(cfg.n, params.clone());
        let config_grind = WhirConfig::<EF, F, GpuChallenger>::new(cfg.n, params);

        let cpu_dft = CpuDft::new(1 << config.max_fft_size());
        let gpu_dft = MetalBabyBearDft::default();
        let cpu_mmcs = CpuMmcs::new(hash.clone(), compress.clone(), 0);
        let gpu_mmcs = GpuMmcsType::new(CpuMmcs::new(hash, compress, 0), gpu_dft.clone());

        let cpu = CpuPcs::new(config.clone(), cpu_dft, cpu_mmcs);
        let gpu = GpuPcs::new(config, gpu_dft.clone(), gpu_mmcs.clone());
        let gpu_grind = GpuGrindPcs::new(config_grind, gpu_dft.clone(), gpu_mmcs);

        let mut data_rng = SmallRng::seed_from_u64(0xD157A1B);
        let table = Table::new(vec![Poly::<F>::rand(&mut data_rng, cfg.n)]);
        let witness = WhirLayout::new_witness(vec![table], cfg.fold);

        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(cfg.n, 1),
            vec![vec![0]; NUM_EVALUATIONS],
        )]);

        let mut domain_separator = DomainSeparator::<EF, F>::new(vec![]);
        cpu.add_domain_separator::<8>(&mut domain_separator);

        Self {
            cpu,
            gpu,
            gpu_grind,
            witness,
            protocol,
            domain_separator,
            perm,
            gpu_dft,
        }
    }

    fn duplex_challenger(&self) -> Duplex {
        let mut c = Duplex::new(self.perm.clone());
        self.domain_separator.observe_domain_separator(&mut c);
        c
    }

    fn gpu_challenger(&self) -> GpuChallenger {
        let mut c = GpuChallenger::new(self.perm.clone(), self.gpu_dft.clone());
        self.domain_separator.observe_domain_separator(&mut c);
        c
    }
}

struct Timings {
    cpu: f64,
    gpu: f64,
    gpu_fused: f64,
    gpu_grind: f64,
}

impl Timings {
    fn best_gpu(&self) -> f64 {
        self.gpu.min(self.gpu_fused).min(self.gpu_grind)
    }

    fn best_mode(&self) -> &'static str {
        let best = self.best_gpu();
        if (best - self.gpu).abs() < 0.01 {
            "gpu"
        } else if (best - self.gpu_fused).abs() < 0.01 {
            "fused"
        } else {
            "grind"
        }
    }

    fn speedup(&self) -> f64 {
        self.cpu / self.best_gpu()
    }
}

fn time_cpu(fx: &Fixture) -> f64 {
    let mut samples = [Duration::ZERO; RUNS];
    for s in &mut samples {
        let mut ch = fx.duplex_challenger();
        let t0 = Instant::now();
        let (_, pd) = <CpuPcs as MultilinearPcs<EF, Duplex>>::commit(
            &fx.cpu,
            fx.witness.clone(),
            &mut ch,
        );
        let _ = <CpuPcs as MultilinearPcs<EF, Duplex>>::open(&fx.cpu, pd, fx.protocol.clone(), &mut ch);
        *s = t0.elapsed();
    }
    median_ms(&mut samples)
}

fn time_gpu(fx: &Fixture) -> f64 {
    let mut samples = [Duration::ZERO; RUNS];
    for s in &mut samples {
        let mut ch = fx.duplex_challenger();
        let t0 = Instant::now();
        let (_, pd) = <GpuPcs as MultilinearPcs<EF, Duplex>>::commit(
            &fx.gpu,
            fx.witness.clone(),
            &mut ch,
        );
        let _ = <GpuPcs as MultilinearPcs<EF, Duplex>>::open(&fx.gpu, pd, fx.protocol.clone(), &mut ch);
        *s = t0.elapsed();
    }
    median_ms(&mut samples)
}

fn time_gpu_fused(fx: &Fixture) -> f64 {
    let mut samples = [Duration::ZERO; RUNS];
    for s in &mut samples {
        let mut ch = fx.duplex_challenger();
        let t0 = Instant::now();
        let (_, pd) = fx.gpu.commit_fused(fx.witness.clone(), &mut ch);
        let _ = fx.gpu.open_fused(pd, fx.protocol.clone(), &mut ch);
        *s = t0.elapsed();
    }
    median_ms(&mut samples)
}

fn time_gpu_grind(fx: &Fixture) -> f64 {
    let mut samples = [Duration::ZERO; RUNS];
    for s in &mut samples {
        let mut ch = fx.gpu_challenger();
        let t0 = Instant::now();
        let (_, pd) = fx.gpu_grind.commit_fused(fx.witness.clone(), &mut ch);
        let _ = fx.gpu_grind.open_fused(pd, fx.protocol.clone(), &mut ch);
        *s = t0.elapsed();
    }
    median_ms(&mut samples)
}

fn run_config(cfg: GridConfig) -> Result<Timings, String> {
    let fx = Fixture::new(cfg);
    Ok(Timings {
        cpu: time_cpu(&fx),
        gpu: time_gpu(&fx),
        gpu_fused: time_gpu_fused(&fx),
        gpu_grind: time_gpu_grind(&fx),
    })
}

struct Row {
    cfg: GridConfig,
    t: Option<Timings>,
    err: Option<String>,
}

fn print_ethresearch_table(rows: &[Row], n: usize) {
    println!("\n#### n={n} (1M+ coefficients at n=20; ethresear.ch M1 grid format)\n");
    println!("| fold | rate | CPU (ms) | Best GPU (ms) | Speedup | mode | gpu | fused | grind |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for row in rows.iter().filter(|r| r.cfg.n == n) {
        match &row.t {
            Some(t) => println!(
                "| {} | {} | {:.1} | {:.1} | {:.2}x | {} | {:.1} | {:.1} | {:.1} |",
                row.cfg.fold,
                row.cfg.rate,
                t.cpu,
                t.best_gpu(),
                t.speedup(),
                t.best_mode(),
                t.gpu,
                t.gpu_fused,
                t.gpu_grind,
            ),
            None => println!(
                "| {} | {} | ERR | ERR | — | {} |",
                row.cfg.fold,
                row.cfg.rate,
                row.err.as_deref().unwrap_or("?"),
            ),
        }
    }
}

/// M1 reference cells from ethresear.ch (for quick visual comparison).
fn print_post_reference() {
    println!("\n### ethresear.ch M1 reference (post, not re-measured here)\n");
    println!("| n | fold | rate | CPU (ms) | Best GPU (ms) | Speedup |");
    println!("| --- | --- | --- | --- | --- | --- |");
    let refs = [
        (20, 1, 1, 267.0, 171.0),
        (20, 2, 1, 127.0, 75.0),
        (22, 1, 1, 1174.0, 579.0),
        (22, 1, 2, 1938.0, 1092.0),
        (22, 2, 1, 441.0, 287.0),
        (22, 4, 1, 166.0, 128.0),
        (24, 1, 1, 4153.0, 2463.0),
        (24, 2, 1, 1814.0, 1049.0),
    ];
    for (n, fold, rate, cpu, gpu) in refs {
        println!(
            "| {n} | {fold} | {rate} | {cpu:.0} | {gpu:.0} | {:.2}x |",
            cpu / gpu
        );
    }
}

fn main() {
    println!("BabyBear WHIR — ethresear.ch-style grid (Plonky3, PrefixProver)");
    println!("security_level={SECURITY_LEVEL}, pow_bits={POW_BITS}, {RUNS} runs, median");
    println!("Best GPU = min(gpu, gpu_fused, gpu_grind)\n");

    let configs = grid_configs();
    let mut rows = Vec::with_capacity(configs.len());

    for (idx, cfg) in configs.iter().enumerate() {
        eprintln!(
            "[{}/{}] n={} fold={} rate={}",
            idx + 1,
            configs.len(),
            cfg.n,
            cfg.fold,
            cfg.rate
        );

        let row = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_config(*cfg))) {
            Ok(Ok(t)) => {
                eprintln!(
                    "  CPU {:.1} | gpu {:.1} | fused {:.1} | grind {:.1} | best {:.1} ({}) {:.2}x",
                    t.cpu,
                    t.gpu,
                    t.gpu_fused,
                    t.gpu_grind,
                    t.best_gpu(),
                    t.best_mode(),
                    t.speedup(),
                );
                Row {
                    cfg: *cfg,
                    t: Some(t),
                    err: None,
                }
            }
            Ok(Err(e)) => Row {
                cfg: *cfg,
                t: None,
                err: Some(e),
            },
            Err(_) => Row {
                cfg: *cfg,
                t: None,
                err: Some("panic".into()),
            },
        };
        rows.push(row);
    }

    for &n in &[20, 22, 24] {
        print_ethresearch_table(&rows, n);
    }

    print_post_reference();

    let ok: Vec<&Timings> = rows.iter().filter_map(|r| r.t.as_ref()).collect();
    if !ok.is_empty() {
        let speeds: Vec<f64> = ok.iter().map(|t| t.speedup()).collect();
        let min = speeds.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = speeds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg = speeds.iter().sum::<f64>() / speeds.len() as f64;
        println!("\n### Summary ({}/{} configs OK)\n", ok.len(), rows.len());
        println!("| metric | value |");
        println!("| --- | --- |");
        println!("| speedup min | {min:.2}x |");
        println!("| speedup max | {max:.2}x |");
        println!("| speedup mean | {avg:.2}x |");
    }
}
