//! Full WHIR CPU vs GPU grid benchmark (BabyBear + KoalaBear).
//!
//! Matches the ethresear.ch parameter grid: `(n, fold, rate)` with 3-run medians
//! for end-to-end proof (commit + open). GPU path uses `commit_fused` / `open_fused`.
//!
//! Run:
//!   `cargo bench -p p3-whir --bench whir_grid_gpu --features gpu-metal --release`
//!
//! Published results: see `BENCHMARKS.md` in this directory.

use std::time::{Duration, Instant};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_dft_metal::{GpuKoalaMmcs, GpuMmcs, MetalBabyBearDft, MetalKoalaBearDft};
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
};
use p3_whir::pcs::prover::WhirProver;
use p3_whir::sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
use p3_whir::sumcheck::{OpeningProtocol, TableShape, TableSpec};
use rand::SeedableRng;
use rand::rngs::SmallRng;

const NUM_EVALUATIONS: usize = 1;
const POW_BITS: usize = 0;
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

fn speedup(cpu: f64, gpu: f64) -> f64 {
    cpu / gpu
}

struct Row {
    n: usize,
    fold: usize,
    rate: usize,
    bb_cpu: Option<f64>,
    bb_gpu: Option<f64>,
    kb_cpu: Option<f64>,
    kb_gpu: Option<f64>,
    bb_err: Option<String>,
    kb_err: Option<String>,
}

mod babybear {
    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MerkleHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MerkleCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type CpuMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
    type CpuDft = Radix2DFTSmallBatch<F>;
    type GpuMmcsType = GpuMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
    type WhirLayout = PrefixProver<F, EF>;
    type CpuPcs = WhirProver<EF, F, CpuDft, CpuMmcs, Challenger, WhirLayout>;
    type GpuPcs = WhirProver<EF, F, MetalBabyBearDft, GpuMmcsType, Challenger, WhirLayout>;

    struct Fixture {
        cpu: CpuPcs,
        gpu: GpuPcs,
        witness: <CpuPcs as MultilinearPcs<EF, Challenger>>::Witness,
        protocol: OpeningProtocol,
        domain_separator: DomainSeparator<EF, F>,
        challenger: Challenger,
    }

    impl Fixture {
        fn new(cfg: GridConfig) -> Self {
            let mut perm_rng = SmallRng::seed_from_u64(1);
            let perm = Perm::new_from_rng_128(&mut perm_rng);
            let hash = MerkleHash::new(perm.clone());
            let compress = MerkleCompress::new(perm.clone());
            let folding_factor = FoldingFactor::Constant(cfg.fold);
            let params = ProtocolParameters {
                security_level: 100,
                pow_bits: POW_BITS,
                round_log_inv_rates: round_log_inv_rates(cfg.n, &folding_factor, cfg.rate),
                folding_factor,
                soundness_type: SecurityAssumption::CapacityBound,
                starting_log_inv_rate: cfg.rate,
            };
            let config = WhirConfig::<EF, F, Challenger>::new(cfg.n, params);

            let cpu_dft = CpuDft::new(1 << config.max_fft_size());
            let gpu_dft = MetalBabyBearDft::default();
            let cpu_mmcs = CpuMmcs::new(hash.clone(), compress.clone(), 0);
            let gpu_mmcs = GpuMmcsType::new(CpuMmcs::new(hash, compress, 0), gpu_dft.clone());

            let cpu = CpuPcs::new(config.clone(), cpu_dft, cpu_mmcs);
            let gpu = GpuPcs::new(config, gpu_dft, gpu_mmcs);

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
                witness,
                protocol,
                domain_separator,
                challenger: Challenger::new(perm),
            }
        }

        fn challenger(&self) -> Challenger {
            let mut c = self.challenger.clone();
            self.domain_separator.observe_domain_separator(&mut c);
            c
        }
    }

    pub fn run(cfg: GridConfig) -> Result<(f64, f64), String> {
        let fx = Fixture::new(cfg);
        let mut cpu_samples = [Duration::ZERO; RUNS];
        for sample in &mut cpu_samples {
            let mut ch = fx.challenger();
            let t0 = Instant::now();
            let (_, pd) = <CpuPcs as MultilinearPcs<EF, Challenger>>::commit(
                &fx.cpu,
                fx.witness.clone(),
                &mut ch,
            );
            let _ = <CpuPcs as MultilinearPcs<EF, Challenger>>::open(
                &fx.cpu,
                pd,
                fx.protocol.clone(),
                &mut ch,
            );
            *sample = t0.elapsed();
        }

        let mut gpu_samples = [Duration::ZERO; RUNS];
        for sample in &mut gpu_samples {
            let mut ch = fx.challenger();
            let t0 = Instant::now();
            let (_, pd) = fx.gpu.commit_fused(fx.witness.clone(), &mut ch);
            let _ = fx.gpu.open_fused(pd, fx.protocol.clone(), &mut ch);
            *sample = t0.elapsed();
        }

        Ok((median_ms(&mut cpu_samples), median_ms(&mut gpu_samples)))
    }
}

mod koalabear {
    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type Poseidon16 = Poseidon2KoalaBear<16>;
    type Poseidon24 = Poseidon2KoalaBear<24>;
    type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
    type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
    type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type CpuMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
    type CpuDft = Radix2DFTSmallBatch<F>;
    type GpuMmcsType = GpuKoalaMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
    type WhirLayout = SuffixProver<F, EF>;
    type CpuPcs = WhirProver<EF, F, CpuDft, CpuMmcs, Challenger, WhirLayout>;
    type GpuPcs = WhirProver<EF, F, MetalKoalaBearDft, GpuMmcsType, Challenger, WhirLayout>;

    struct Fixture {
        cpu: CpuPcs,
        gpu: GpuPcs,
        witness: <CpuPcs as MultilinearPcs<EF, Challenger>>::Witness,
        protocol: OpeningProtocol,
        domain_separator: DomainSeparator<EF, F>,
        challenger: Challenger,
    }

    impl Fixture {
        fn new(cfg: GridConfig) -> Self {
            let mut perm_rng = SmallRng::seed_from_u64(1);
            let poseidon16 = Poseidon16::new_from_rng_128(&mut perm_rng);
            let poseidon24 = Poseidon24::new_from_rng_128(&mut perm_rng);
            let hash = MerkleHash::new(poseidon24);
            let compress = MerkleCompress::new(poseidon16.clone());
            let folding_factor = FoldingFactor::Constant(cfg.fold);
            let params = ProtocolParameters {
                // Match BabyBear grid (128 trips KoalaBear `grind` when folding PoW ≥ 31 bits).
                security_level: 100,
                pow_bits: POW_BITS,
                round_log_inv_rates: round_log_inv_rates(cfg.n, &folding_factor, cfg.rate),
                folding_factor,
                soundness_type: SecurityAssumption::CapacityBound,
                starting_log_inv_rate: cfg.rate,
            };
            let config = WhirConfig::<EF, F, Challenger>::new(cfg.n, params);

            let cpu_dft = CpuDft::new(1 << config.max_fft_size());
            let gpu_dft = MetalKoalaBearDft::default();
            let cpu_mmcs = CpuMmcs::new(hash.clone(), compress.clone(), 0);
            let gpu_mmcs = GpuMmcsType::new(CpuMmcs::new(hash, compress, 0), gpu_dft.clone());

            let cpu = CpuPcs::new(config.clone(), cpu_dft, cpu_mmcs);
            let gpu = GpuPcs::new(config, gpu_dft, gpu_mmcs);

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
                witness,
                protocol,
                domain_separator,
                challenger: Challenger::new(poseidon16),
            }
        }

        fn challenger(&self) -> Challenger {
            let mut c = self.challenger.clone();
            self.domain_separator.observe_domain_separator(&mut c);
            c
        }
    }

    pub fn run(cfg: GridConfig) -> Result<(f64, f64), String> {
        let fx = Fixture::new(cfg);
        let mut cpu_samples = [Duration::ZERO; RUNS];
        for sample in &mut cpu_samples {
            let mut ch = fx.challenger();
            let t0 = Instant::now();
            let (_, pd) = <CpuPcs as MultilinearPcs<EF, Challenger>>::commit(
                &fx.cpu,
                fx.witness.clone(),
                &mut ch,
            );
            let _ = <CpuPcs as MultilinearPcs<EF, Challenger>>::open(
                &fx.cpu,
                pd,
                fx.protocol.clone(),
                &mut ch,
            );
            *sample = t0.elapsed();
        }

        let mut gpu_samples = [Duration::ZERO; RUNS];
        for sample in &mut gpu_samples {
            let mut ch = fx.challenger();
            let t0 = Instant::now();
            let (_, pd) = fx.gpu.commit_fused(fx.witness.clone(), &mut ch);
            let _ = fx.gpu.open_fused(pd, fx.protocol.clone(), &mut ch);
            *sample = t0.elapsed();
        }

        Ok((median_ms(&mut cpu_samples), median_ms(&mut gpu_samples)))
    }
}

fn print_table(rows: &[Row], n: usize) {
    println!("\n### n={n} (full proof, gpu_fused, PoW=0, 3-run median)\n");
    println!(
        "| fold | rate | BB CPU (ms) | BB GPU (ms) | BB speedup | KB CPU (ms) | KB GPU (ms) | KB speedup |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    for row in rows.iter().filter(|r| r.n == n) {
        let fmt = |v: Option<f64>| v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "—".into());
        let fmt_sp = |cpu: Option<f64>, gpu: Option<f64>| {
            match (cpu, gpu) {
                (Some(c), Some(g)) => format!("{:.2}x", speedup(c, g)),
                _ => "—".into(),
            }
        };
        let bb_fail = row.bb_err.is_some();
        let kb_fail = row.kb_err.is_some();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            row.fold,
            row.rate,
            if bb_fail { "ERR".into() } else { fmt(row.bb_cpu) },
            if bb_fail { "ERR".into() } else { fmt(row.bb_gpu) },
            if bb_fail {
                row.bb_err.clone().unwrap_or_default()
            } else {
                fmt_sp(row.bb_cpu, row.bb_gpu)
            },
            if kb_fail { "ERR".into() } else { fmt(row.kb_cpu) },
            if kb_fail { "ERR".into() } else { fmt(row.kb_gpu) },
            if kb_fail {
                row.kb_err.clone().unwrap_or_default()
            } else {
                fmt_sp(row.kb_cpu, row.kb_gpu)
            },
        );
    }
}

fn main() {
    println!("WHIR grid benchmark: BabyBear (prefix) vs KoalaBear (suffix)");
    println!("GPU path: commit_fused + open_fused only (not gpu_grind)");
    println!("Configs: {} cells (ethresear.ch M1 grid)\n", grid_configs().len());

    let only_koala = std::env::var("WHIR_GRID_KOALA_ONLY").is_ok();
    let only_babybear = std::env::var("WHIR_GRID_BABYBEAR_ONLY").is_ok();

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

        let mut row = Row {
            n: cfg.n,
            fold: cfg.fold,
            rate: cfg.rate,
            bb_cpu: None,
            bb_gpu: None,
            kb_cpu: None,
            kb_gpu: None,
            bb_err: None,
            kb_err: None,
        };

        if !only_koala {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| babybear::run(*cfg))) {
            Ok(Ok((cpu, gpu))) => {
                row.bb_cpu = Some(cpu);
                row.bb_gpu = Some(gpu);
                eprintln!(
                    "  BabyBear  CPU {:.1} ms  GPU {:.1} ms  {:.2}x",
                    cpu,
                    gpu,
                    speedup(cpu, gpu)
                );
            }
            Ok(Err(e)) => {
                row.bb_err = Some(e);
                eprintln!("  BabyBear  ERROR");
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| e.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic".into());
                row.bb_err = Some(msg.clone());
                eprintln!("  BabyBear  PANIC: {msg}");
            }
        }
        }

        if !only_babybear {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| koalabear::run(*cfg))) {
            Ok(Ok((cpu, gpu))) => {
                row.kb_cpu = Some(cpu);
                row.kb_gpu = Some(gpu);
                eprintln!(
                    "  KoalaBear CPU {:.1} ms  GPU {:.1} ms  {:.2}x",
                    cpu,
                    gpu,
                    speedup(cpu, gpu)
                );
            }
            Ok(Err(e)) => {
                row.kb_err = Some(e);
                eprintln!("  KoalaBear ERROR");
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| e.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic".into());
                row.kb_err = Some(msg.clone());
                eprintln!("  KoalaBear PANIC: {msg}");
            }
        }
        }

        rows.push(row);
    }

    for &n in &[20, 22, 24] {
        print_table(&rows, n);
    }

    let bb_speedups: Vec<f64> = rows
        .iter()
        .filter_map(|r| match (r.bb_cpu, r.bb_gpu) {
            (Some(c), Some(g)) => Some(speedup(c, g)),
            _ => None,
        })
        .collect();
    let kb_speedups: Vec<f64> = rows
        .iter()
        .filter_map(|r| match (r.kb_cpu, r.kb_gpu) {
            (Some(c), Some(g)) => Some(speedup(c, g)),
            _ => None,
        })
        .collect();

    if !bb_speedups.is_empty() && !kb_speedups.is_empty() {
        let bb_min = bb_speedups.iter().cloned().fold(f64::INFINITY, f64::min);
        let bb_max = bb_speedups.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let kb_min = kb_speedups.iter().cloned().fold(f64::INFINITY, f64::min);
        let kb_max = kb_speedups.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bb_avg = bb_speedups.iter().sum::<f64>() / bb_speedups.len() as f64;
        let kb_avg = kb_speedups.iter().sum::<f64>() / kb_speedups.len() as f64;
        println!("\n### Summary\n");
        println!("| Field | configs OK | speedup min | speedup max | speedup mean |");
        println!("| --- | --- | --- | --- | --- |");
        println!(
            "| BabyBear | {} | {:.2}x | {:.2}x | {:.2}x |",
            bb_speedups.len(),
            bb_min,
            bb_max,
            bb_avg
        );
        println!(
            "| KoalaBear | {} | {:.2}x | {:.2}x | {:.2}x |",
            kb_speedups.len(),
            kb_min,
            kb_max,
            kb_avg
        );
    }
}
