//! Base-field commitment used by the sumcheck opening protocol.

use p3_challenger::CanObserve;
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_multilinear_util::poly::Poly;
use tracing::info_span;

use crate::strategy::VariableOrder;

/// Encodes and commits the initial base-field polynomial.
///
/// This is the first WHIR commitment. It lays out the polynomial according to
/// the residual variable order, applies the Reed-Solomon expansion with `dft`,
/// commits the resulting codeword matrix with `mmcs`, and observes the Merkle
/// root in the transcript.
///
/// Prefix order transposes the local folding block before padding so the first
/// folded variables become columns. Suffix order keeps the folding block as the
/// row width and only zero-pads the row count.
pub fn commit_base<F, Dft, MT, Challenger>(
    order: VariableOrder,
    dft: &Dft,
    mmcs: &MT,
    challenger: &mut Challenger,
    poly: &Poly<F>,
    folding: usize,
    starting_log_inv_rate: usize,
) -> (MT::Commitment, MT::ProverData<DenseMatrix<F>>)
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: CanObserve<MT::Commitment>,
{
    let num_variables = poly.num_variables();
    let height = 1 << (num_variables + starting_log_inv_rate - folding);

    let encoded = match order {
        VariableOrder::Prefix => {
            let padded = info_span!("transpose & pad").in_scope(|| {
                let mut mat =
                    RowMajorMatrixView::new(poly.as_slice(), 1 << (num_variables - folding))
                        .transpose();
                mat.pad_to_height(height, F::ZERO);
                mat
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_batch(padded).to_row_major_matrix())
        }
        VariableOrder::Suffix => {
            let padded = info_span!("pad").in_scope(|| {
                let mut mat = RowMajorMatrix::new(poly.as_slice().to_vec(), 1 << folding);
                mat.pad_to_height(height, F::ZERO);
                mat
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_batch(padded).to_row_major_matrix())
        }
    };

    let (root, prover_data) = info_span!("commit_matrix").in_scope(|| mmcs.commit_matrix(encoded));
    challenger.observe(root.clone());
    (root, prover_data)
}

/// Like [`commit_base`], but attempts GPU-fused transpose+pad+DFT+Merkle when supported.
#[cfg(feature = "gpu-metal")]
pub fn commit_base_fused<F, Dft, MT, Challenger>(
    order: VariableOrder,
    dft: &Dft,
    mmcs: &MT,
    challenger: &mut Challenger,
    poly: &Poly<F>,
    folding: usize,
    starting_log_inv_rate: usize,
) -> (MT::Commitment, MT::ProverData<DenseMatrix<F>>)
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F> + p3_dft_metal::DftCommitFusion<F>,
    Challenger: CanObserve<MT::Commitment>,
{
    let num_variables = poly.num_variables();
    let in_cols = 1 << (num_variables - folding);
    let in_rows = 1 << folding;
    let padded_height = 1 << (num_variables + starting_log_inv_rate - folding);

    let (root, prover_data) = match order {
        VariableOrder::Prefix => {
            if let Some(result) = info_span!("fused_transpose_dft_commit").in_scope(|| {
                mmcs.transpose_pad_dft_and_commit(
                    poly.as_slice(),
                    in_rows,
                    in_cols,
                    padded_height,
                )
            }) {
                result
            } else {
                commit_base(
                    order,
                    dft,
                    mmcs,
                    challenger,
                    poly,
                    folding,
                    starting_log_inv_rate,
                )
            }
        }
        VariableOrder::Suffix => {
            let padded = info_span!("pad").in_scope(|| {
                let mut mat = RowMajorMatrix::new(poly.as_slice().to_vec(), 1 << folding);
                mat.pad_to_height(padded_height, F::ZERO);
                mat
            });
            match info_span!("fused_dft_commit").in_scope(|| mmcs.dft_and_commit(padded)) {
                Ok((root, tree)) => (root, tree),
                Err(padded) => {
                    let encoded = info_span!("dft", height = padded.height(), width = padded.width())
                        .in_scope(|| dft.dft_batch(padded).to_row_major_matrix());
                    info_span!("commit_matrix").in_scope(|| mmcs.commit_matrix(encoded))
                }
            }
        }
    };

    challenger.observe(root.clone());
    (root, prover_data)
}
