//! WHIR PCS smoke test with GPU Metal (macOS/iOS).
//!
//! `cargo run -p p3-whir --example whir_gpu --features gpu-metal --release`

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_whir_metal::{GpuMmcs, MetalBabyBearDft};
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
};
use p3_whir::pcs::prover::WhirProver;
use p3_whir::sumcheck::layout::{Layout as _, PrefixProver, Table};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MerkleHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MerkleCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type PackedF = <F as Field>::Packing;
type ValMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type GpuMmcsType = GpuMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type Layout = PrefixProver<F, EF>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type MyPcs = WhirProver<EF, F, MetalBabyBearDft, GpuMmcsType, MyChallenger, Layout>;

fn main() {
    let num_variables = 18;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MerkleHash::new(perm.clone());
    let compress = MerkleCompress::new(perm.clone());
    let mmcs_cpu = ValMmcs::new(hash, compress, 0);
    let dft = MetalBabyBearDft::default();
    let mmcs = GpuMmcsType::new(mmcs_cpu, dft.clone());
    let config = WhirConfig::new(
        num_variables,
        ProtocolParameters {
            security_level: 100,
            pow_bits: 0,
            folding_factor: FoldingFactor::Constant(4),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            round_log_inv_rates: vec![],
        },
    );

    let pcs = MyPcs::new(config, dft, mmcs);

    let poly = Poly::<F>::new((0..1 << num_variables).map(|_| rng.random()).collect());
    let witness = Layout::new_witness(vec![Table::new(vec![poly])], 4);

    let mut challenger = MyChallenger::new(perm);
    let (_commitment, _prover_data) = pcs.commit_fused(witness, &mut challenger);
    println!("GPU-fused WHIR commit ok (n={num_variables})");
}
