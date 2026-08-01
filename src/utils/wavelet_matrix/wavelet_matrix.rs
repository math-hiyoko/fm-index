use std::{collections, hash, iter, ops};

use num_traits::{One, PrimInt, Unsigned, Zero};
use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::bit_vector::BitVector;
use crate::utils::traits::{
    bit_vector::BlockType, bit_width::BitWidth, wavelet_matrix::WaveletMatrixTrait,
};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct WaveletMatrix<NumberType>
where
    NumberType: PrimInt + Unsigned + hash::Hash,
{
    len: usize,
    height: usize,
    layers: Vec<BitVector>,
    zeros_count_per_layer: Vec<usize>,
    begin_index: collections::HashMap<NumberType, usize>,
}

impl<NumberType> WaveletMatrix<NumberType>
where
    NumberType: PrimInt + Unsigned + BitWidth + hash::Hash + Send + Sync,
{
    pub(crate) fn new(mut values: Vec<NumberType>) -> PyResult<Self> {
        let len = values.len();

        let height = values
            .par_iter()
            .max()
            .map_or(0usize, |max| max.bit_width());

        let mut zeros_count_per_layer = Vec::with_capacity(height);
        let mut layer_blocks_vec = Vec::with_capacity(height);
        for depth in 0..height {
            let current_layer_bits = values
                .par_iter()
                .map(|&value| (value >> (height - depth - 1) & NumberType::one()).is_one())
                .collect::<Vec<_>>();
            let zeros_count = current_layer_bits.par_iter().filter(|&&b| !b).count();

            let mut next_values = vec![NumberType::zero(); len];
            let mut zero_index = 0usize;
            let mut one_index = zeros_count;
            for (&bit, value) in iter::zip(&current_layer_bits, values) {
                if bit {
                    next_values[one_index] = value;
                    one_index += 1;
                } else {
                    next_values[zero_index] = value;
                    zero_index += 1;
                }
            }

            let current_layer_blocks = current_layer_bits
                .into_par_iter()
                .chunks(BlockType::BITS as usize)
                .map(|chunk| {
                    chunk
                        .iter()
                        .enumerate()
                        .fold(BlockType::zero(), |acc, (j, &b)| {
                            if b {
                                acc | (BlockType::one() << j)
                            } else {
                                acc
                            }
                        })
                })
                .collect::<Vec<_>>();

            zeros_count_per_layer.push(zeros_count);
            layer_blocks_vec.push(current_layer_blocks);
            values = next_values;
        }

        let mut begin_index = collections::HashMap::new();
        for (position, value) in values.into_iter().enumerate() {
            begin_index.entry(value).or_insert(position);
        }

        let layers = layer_blocks_vec
            .into_par_iter()
            .map(|blocks| BitVector::new(blocks, len))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(WaveletMatrix {
            len,
            height,
            layers,
            zeros_count_per_layer,
            begin_index,
        })
    }
}

impl<NumberType> WaveletMatrixTrait<NumberType, BitVector> for WaveletMatrix<NumberType>
where
    NumberType: PrimInt
        + Unsigned
        + BitWidth
        + hash::Hash
        + ops::BitOrAssign
        + ops::ShlAssign<usize>
        + Send
        + Sync,
{
    fn len(&self) -> usize {
        self.len
    }

    fn max_bit(&self) -> usize {
        self.height
    }

    fn get_layers(&self) -> &[BitVector] {
        &self.layers
    }

    fn get_zeros_count_per_layer(&self) -> &[usize] {
        &self.zeros_count_per_layer
    }

    fn begin_index(&self, value: NumberType) -> Option<usize> {
        self.begin_index.get(&value).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn create_test_wavelet_matrix() -> WaveletMatrix<u32> {
        let test_data = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        WaveletMatrix::new(test_data).unwrap()
    }

    #[test]
    fn test_empty_wavelet_matrix() {
        Python::initialize();

        let wm = WaveletMatrix::<u32>::new(vec![]).unwrap();

        // Access should fail on empty matrix
        assert_eq!(
            wm.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );

        // Values should return empty vector
        assert_eq!(wm.values().unwrap(), Vec::<u32>::new());

        // Rank should return 0
        assert_eq!(wm.rank(0u32, 0).unwrap(), 0);

        // Select with kth=0 should fail
        assert_eq!(
            wm.select(0u32, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );

        // Range list should fail for out of bounds
        assert_eq!(
            wm.range_list(0, 1).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
    }

    #[test]
    fn test_all_zeros() {
        Python::initialize();

        let wm = WaveletMatrix::new(vec![0u32; 64]).unwrap();

        assert_eq!(wm.access(1).unwrap(), 0u32);
        assert_eq!(wm.values().unwrap(), vec![0u32; 64]);
        assert_eq!(wm.rank(0u32, 1).unwrap(), 1);
        assert_eq!(wm.select(0u32, 1).unwrap(), Some(0));

        // Range list should show all 64 values are 0
        let result = wm.range_list(0, 64).unwrap();
        assert_eq!(result, vec![(0u32, 64)]);

        // Partial range
        let result = wm.range_list(10, 20).unwrap();
        assert_eq!(result, vec![(0u32, 10)]);
    }

    #[test]
    fn test_all_zero_values() {
        Python::initialize();

        let wm = WaveletMatrix::new(vec![0u32; 64]).unwrap();

        assert_eq!(wm.access(1).unwrap(), 0u32);
        assert_eq!(wm.values().unwrap(), vec![0u32; 64]);
        assert_eq!(wm.rank(0u32, 1).unwrap(), 1);
        assert_eq!(wm.select(0u32, 1).unwrap(), Some(0));

        // Range list should show all 64 values are 0
        assert_eq!(wm.range_list(0, 64).unwrap(), vec![(0u32, 64)]);
    }

    #[test]
    fn test_maximum_value() {
        Python::initialize();

        let wm = WaveletMatrix::new(vec![u32::MAX - 1; 64]).unwrap();

        assert_eq!(wm.access(1).unwrap(), u32::MAX - 1);
        assert_eq!(wm.values().unwrap(), vec![u32::MAX - 1; 64]);
        assert_eq!(wm.rank(u32::MAX - 1, 1).unwrap(), 1);
        assert_eq!(wm.select(u32::MAX - 1, 1).unwrap(), Some(0));

        // Range list should show all 64 values are u32::MAX - 1
        assert_eq!(
            wm.range_list(0, wm.len()).unwrap(),
            vec![(u32::MAX - 1, 64)]
        );

        // Partial range
        assert_eq!(wm.range_list(5, 15).unwrap(), vec![(u32::MAX - 1, 10)]);
    }

    #[test]
    fn test_access_operation() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        assert_eq!(wm.access(6).unwrap(), 5u32);
        assert_eq!(wm.access(7).unwrap(), 6u32);

        // Range list for small range containing these values
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        // Indices [6, 8): [5, 6]
        assert_eq!(wm.range_list(6, 8).unwrap(), vec![(5u32, 1), (6u32, 1)]);
    }

    #[test]
    fn test_values_retrieval() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        let expected = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        assert_eq!(wm.values().unwrap(), expected);

        // Range list should match the count of values
        assert_eq!(
            wm.range_list(0, 12).unwrap(),
            vec![(0u32, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 5), (6, 1)]
        );
    }

    #[test]
    fn test_rank_operation() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        assert_eq!(wm.rank(5u32, 11).unwrap(), 5);
        assert_eq!(wm.rank(1u32, 11).unwrap(), 2);

        // Range list should match rank counts in range
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        // Indices [0, 11): [5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5] (excluding last element 0)
        assert_eq!(
            wm.range_list(0, 11).unwrap(),
            vec![(1u32, 2), (2, 1), (3, 1), (4, 1), (5, 5), (6, 1)]
        );
    }

    #[test]
    fn test_select_operation() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        // Valid selections
        assert_eq!(wm.select(5u32, 4).unwrap(), Some(6));
        assert_eq!(wm.select(1u32, 2).unwrap(), Some(8));

        // Out of range selections
        assert_eq!(wm.select(5u32, 6).unwrap(), None);
        assert_eq!(wm.select(1u32, 6).unwrap(), None);

        // Range list around selected positions
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        // 4th occurrence of 5 is at index 6, 2nd occurrence of 1 is at index 8
        // Range [6, 9): [5, 6, 1]
        assert_eq!(
            wm.range_list(6, 9).unwrap(),
            vec![(1u32, 1), (5, 1), (6, 1)]
        );
    }

    #[test]
    fn test_range_list_full_range() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]

        assert_eq!(
            wm.range_list(0, 12).unwrap(),
            vec![(0u32, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 5), (6, 1)],
        );
    }

    #[test]
    fn test_range_list_partial_range() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        // Indices:        0  1  2  3  4  5  6  7  8  9  10 11

        // Test range [0, 5): [5, 4, 5, 5, 2]
        assert_eq!(
            wm.range_list(0, 5).unwrap(),
            vec![(2u32, 1), (4, 1), (5, 3)],
        );

        // Test range [5, 10): [1, 5, 6, 1, 3]
        assert_eq!(
            wm.range_list(5, 10).unwrap(),
            vec![(1u32, 2), (3, 1), (5, 1), (6, 1)],
        );
    }

    #[test]
    fn test_range_list_errors() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        // Error: start > end
        assert_eq!(
            wm.range_list(10, 5).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );

        // Error: end > len
        assert_eq!(
            wm.range_list(0, 13).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
    }

    #[test]
    fn test_topk_basic() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        // Value frequencies in full range: 5->5, 1->2, 0->1, 2->1, 3->1, 4->1, 6->1

        // Top 3 values
        assert_eq!(wm.topk(0, 12, 3).unwrap(), vec![(5u32, 5), (1, 2), (3, 1)]);
    }

    #[test]
    fn test_topk_all_values() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]

        // Request more than unique values (should return all 7 unique values)
        assert_eq!(
            wm.topk(0, 12, 100).unwrap(),
            vec![(5u32, 5), (1, 2), (3, 1), (2, 1), (4, 1), (0, 1), (6, 1)]
        );
    }

    #[test]
    fn test_topk_partial_range() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        // Indices:        0  1  2  3  4  5  6  7  8  9  10 11

        // Range [0, 5): [5, 4, 5, 5, 2]
        assert_eq!(wm.topk(0, 5, 2).unwrap(), vec![(5u32, 3), (4, 1)]); // Top 2 values

        // Range [5, 10): [1, 5, 6, 1, 3]
        assert_eq!(wm.topk(5, 10, 2).unwrap(), vec![(1u32, 2), (6, 1)]); // Top 2 values
    }

    #[test]
    fn test_topk_single_element() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        // Single element range
        assert_eq!(wm.topk(0, 1, 1).unwrap(), vec![(5u32, 1)]); // Only one element (5)
    }

    #[test]
    fn test_topk_all_same_frequency() {
        Python::initialize();

        // All different values with same frequency
        let wm = WaveletMatrix::<u32>::new(vec![1, 2, 3, 4, 5]).unwrap();

        assert_eq!(
            wm.topk(0, 5, 3).unwrap(),
            vec![(3, 1), (5, 1), (2, 1)] // Any three values with frequency 1
        );
    }

    #[test]
    fn test_topk_errors() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        // Error: start >= end
        assert_eq!(
            wm.topk(5, 5, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );

        assert_eq!(
            wm.topk(10, 5, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );

        // Error: end > len
        assert_eq!(
            wm.topk(0, 13, 1).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );

        // Error: k == 0
        assert_eq!(
            wm.topk(0, 12, 0).unwrap_err().to_string(),
            "ValueError: k must be greater than 0"
        );
    }

    #[test]
    fn test_topk_empty_matrix() {
        Python::initialize();

        let wm = WaveletMatrix::<u32>::new(vec![]).unwrap();

        // Empty matrix should fail
        assert_eq!(
            wm.topk(0, 0, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
    }

    #[test]
    fn test_topk_all_zeros() {
        Python::initialize();

        let wm = WaveletMatrix::new(vec![0u32; 64]).unwrap();

        assert_eq!(wm.topk(0, 64, 1).unwrap(), vec![(0u32, 64)]); // All 64 values are 0
        assert_eq!(wm.topk(10, 20, 5).unwrap(), vec![(0u32, 10)]); // Only one unique value (0)
    }

    #[test]
    fn test_topk_large_values() {
        Python::initialize();

        let wm = WaveletMatrix::new(vec![u32::MAX; 10]).unwrap();

        assert_eq!(wm.topk(0, 10, 1).unwrap(), vec![(u32::MAX, 10)]); // All values are u32::MAX
    }
}
