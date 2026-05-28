use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::extension::FlatMatrixView;
use p3_multilinear_util::poly::Poly;
use tracing::info_span;

use crate::sumcheck::strategy::VariableOrder;

/// Encodes and commits a folded extension-field polynomial.
///
/// This is used after each non-final WHIR folding round. The layout mirrors
/// the base-field path, but the DFT runs over extension-field values and the
/// commitment is made through an extension MMCS that views extension rows as
/// base-field data for the underlying Merkle tree.
#[allow(clippy::type_complexity)]
pub(crate) fn commit_extension<F, EF, Dft, MT>(
    order: VariableOrder,
    dft: &Dft,
    extension_mmcs: &ExtensionMmcs<F, EF, MT>,
    poly: &Poly<EF>,
    folding: usize,
    inv_rate: usize,
) -> (
    MT::Commitment,
    <MT as Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
{
    let num_variables = poly.num_variables();
    let height = inv_rate * (1 << (num_variables - folding));

    let encoded = match order {
        VariableOrder::Prefix => {
            let padded = info_span!("transpose & pad").in_scope(|| {
                let mut mat =
                    RowMajorMatrixView::new(poly.as_slice(), 1 << (num_variables - folding))
                        .transpose();
                mat.pad_to_height(height, EF::ZERO);
                mat
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_algebra_batch(padded).to_row_major_matrix())
        }
        VariableOrder::Suffix => {
            let padded = info_span!("pad").in_scope(|| {
                let mut mat = RowMajorMatrix::new(poly.as_slice().to_vec(), 1 << folding);
                mat.pad_to_height(height, EF::ZERO);
                mat
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_algebra_batch(padded).to_row_major_matrix())
        }
    };

    info_span!("commit_matrix").in_scope(|| extension_mmcs.commit_matrix(encoded))
}

/// Like [`commit_extension`], but attempts GPU-fused transpose+pad+DFT+Merkle when supported.
#[cfg(feature = "gpu-metal")]
pub(crate) fn commit_extension_fused<F, EF, Dft, MT>(
    order: VariableOrder,
    dft: &Dft,
    mmcs: &MT,
    poly: &Poly<EF>,
    folding: usize,
    inv_rate: usize,
) -> (
    MT::Commitment,
    <MT as p3_commit::Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + p3_field::BasedVectorSpace<F> + Clone + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    MT: p3_commit::Mmcs<F> + p3_whir_metal::DftCommitFusion<F>,
{
    let num_variables = poly.num_variables();
    let in_cols = 1 << (num_variables - folding);
    let in_rows = 1 << folding;
    let padded_height = inv_rate * in_cols;

    let (root, prover_data) = match order {
        VariableOrder::Prefix => {
            if let Some(result) = info_span!("fused_transpose_dft_algebra_commit").in_scope(|| {
                mmcs.transpose_pad_dft_algebra_and_commit::<EF>(
                    poly.as_slice(),
                    in_rows,
                    in_cols,
                    padded_height,
                )
            }) {
                result
            } else {
                let padded = info_span!("transpose & pad").in_scope(|| {
                    let mut mat =
                        RowMajorMatrixView::new(poly.as_slice(), 1 << (num_variables - folding))
                            .transpose();
                    mat.pad_to_height(padded_height, EF::ZERO);
                    mat
                });
                match info_span!("fused_dft_algebra_commit").in_scope(|| {
                    mmcs.dft_algebra_and_commit(padded)
                }) {
                    Ok((root, tree)) => (root, tree),
                    Err(padded) => {
                        let encoded = info_span!(
                            "dft",
                            height = padded.height(),
                            width = padded.width()
                        )
                        .in_scope(|| dft.dft_algebra_batch(padded).to_row_major_matrix());
                        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
                        info_span!("commit_matrix")
                            .in_scope(|| extension_mmcs.commit_matrix(encoded))
                    }
                }
            }
        }
        VariableOrder::Suffix => {
            let padded = info_span!("pad").in_scope(|| {
                let mut mat = RowMajorMatrix::new(poly.as_slice().to_vec(), 1 << folding);
                mat.pad_to_height(padded_height, EF::ZERO);
                mat
            });
            match info_span!("fused_dft_algebra_commit").in_scope(|| {
                mmcs.dft_algebra_and_commit(padded)
            }) {
                Ok((root, tree)) => (root, tree),
                Err(padded) => {
                    let encoded = info_span!("dft", height = padded.height(), width = padded.width())
                        .in_scope(|| dft.dft_algebra_batch(padded).to_row_major_matrix());
                    let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
                    info_span!("commit_matrix").in_scope(|| extension_mmcs.commit_matrix(encoded))
                }
            }
        }
    };

    (root, prover_data)
}
