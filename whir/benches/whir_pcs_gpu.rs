//! WHIR PCS: CPU vs GPU (Metal) end-to-end comparison on BabyBear.
//!
//! Run:
//!   `cargo bench -p p3-whir --bench whir_pcs_gpu --features gpu-metal -- --noplot`

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_dft_metal::{GpuMmcs, MetalBabyBearDft};
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

const FOLDING: usize = 4;
const LOG_INV_RATE: usize = 1;
const SECURITY_LEVEL: usize = 100;
const NUM_EVALUATIONS: usize = 1;
/// PoW off so timings reflect DFT+Merkle + sumcheck, not grinding.
const POW_BITS: usize = 0;

const SIZES: &[(usize, &str)] = &[(14, "n14"), (18, "n18"), (20, "n20"), (22, "n22")];

fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let (num_rounds, _) = folding_factor.compute_number_of_rounds(num_variables);
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = LOG_INV_RATE;
    for round in 0..num_rounds {
        rate += folding_factor.at_round(round) - 1;
        rates.push(rate);
    }
    rates
}

struct Fixture {
    cpu: CpuPcs,
    gpu: GpuPcs,
    witness: <CpuPcs as MultilinearPcs<EF, Challenger>>::Witness,
    protocol: OpeningProtocol,
    domain_separator: DomainSeparator<EF, F>,
    challenger: Challenger,
}

impl Fixture {
    fn new(num_variables: usize) -> Self {
        let mut perm_rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut perm_rng);
        let hash = MerkleHash::new(perm.clone());
        let compress = MerkleCompress::new(perm.clone());
        let folding_factor = FoldingFactor::Constant(FOLDING);
        let params = ProtocolParameters {
            security_level: SECURITY_LEVEL,
            pow_bits: POW_BITS,
            round_log_inv_rates: default_round_log_inv_rates(num_variables, &folding_factor),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: LOG_INV_RATE,
        };
        let config = WhirConfig::<EF, F, Challenger>::new(num_variables, params);

        let cpu_dft = CpuDft::new(1 << config.max_fft_size());
        let gpu_dft = MetalBabyBearDft::default();
        let cpu_mmcs = CpuMmcs::new(hash.clone(), compress.clone(), 0);
        let gpu_mmcs = GpuMmcsType::new(
            CpuMmcs::new(hash, compress, 0),
            gpu_dft.clone(),
        );

        let cpu = CpuPcs::new(config.clone(), cpu_dft, cpu_mmcs);
        let gpu = GpuPcs::new(config, gpu_dft, gpu_mmcs);

        let mut data_rng = SmallRng::seed_from_u64(0xD157A1B);
        let table = Table::new(vec![Poly::<F>::rand(&mut data_rng, num_variables)]);
        let witness = WhirLayout::new_witness(vec![table], FOLDING);

        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
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

fn configure(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(2));
}

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir_cpu_vs_gpu/commit");
    configure(&mut group);

    for &(n, label) in SIZES {
        let fx = Fixture::new(n);
        group.bench_function(BenchmarkId::new("cpu", label), |b| {
            b.iter_batched(
                || (fx.witness.clone(), fx.challenger()),
                |(witness, mut ch)| {
                    <CpuPcs as MultilinearPcs<EF, Challenger>>::commit(&fx.cpu, witness, &mut ch)
                },
                BatchSize::PerIteration,
            );
        });
        group.bench_function(BenchmarkId::new("gpu_fused", label), |b| {
            b.iter_batched(
                || (fx.witness.clone(), fx.challenger()),
                |(witness, mut ch)| fx.gpu.commit_fused(witness, &mut ch),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir_cpu_vs_gpu/prove");
    configure(&mut group);

    for &(n, label) in SIZES {
        let fx = Fixture::new(n);
        group.bench_function(BenchmarkId::new("cpu", label), |b| {
            b.iter_batched(
                || {
                    let mut ch = fx.challenger();
                    let (_, pd) = <CpuPcs as MultilinearPcs<EF, Challenger>>::commit(
                        &fx.cpu,
                        fx.witness.clone(),
                        &mut ch,
                    );
                    (pd, ch)
                },
                |(pd, mut ch)| {
                    <CpuPcs as MultilinearPcs<EF, Challenger>>::open(
                        &fx.cpu, pd, fx.protocol.clone(), &mut ch,
                    )
                },
                BatchSize::PerIteration,
            );
        });
        group.bench_function(BenchmarkId::new("gpu_fused", label), |b| {
            b.iter_batched(
                || {
                    let mut ch = fx.challenger();
                    let (_, pd) = fx.gpu.commit_fused(fx.witness.clone(), &mut ch);
                    (pd, ch)
                },
                |(pd, mut ch)| fx.gpu.open_fused(pd, fx.protocol.clone(), &mut ch),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn bench_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir_cpu_vs_gpu/full");
    configure(&mut group);

    for &(n, label) in SIZES {
        let fx = Fixture::new(n);
        group.bench_function(BenchmarkId::new("cpu", label), |b| {
            b.iter_batched(
                || (fx.witness.clone(), fx.challenger(), fx.protocol.clone()),
                |(witness, mut ch, protocol)| {
                    let (_, pd) =
                        <CpuPcs as MultilinearPcs<EF, Challenger>>::commit(&fx.cpu, witness, &mut ch);
                    <CpuPcs as MultilinearPcs<EF, Challenger>>::open(&fx.cpu, pd, protocol, &mut ch)
                },
                BatchSize::PerIteration,
            );
        });
        group.bench_function(BenchmarkId::new("gpu_fused", label), |b| {
            b.iter_batched(
                || (fx.witness.clone(), fx.challenger(), fx.protocol.clone()),
                |(witness, mut ch, protocol)| {
                    let (_, pd) = fx.gpu.commit_fused(witness, &mut ch);
                    fx.gpu.open_fused(pd, protocol, &mut ch)
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_commit, bench_prove, bench_full);
criterion_main!(benches);
