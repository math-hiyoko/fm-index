use std::{cmp, collections, iter};

use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyValueError},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    bit_vector::{BitVector, BlockType},
    bit_width::bit_width,
};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct WaveletMatrix {
    len: usize,
    height: usize,
    layers: Vec<BitVector>,
    zeros_count_per_layer: Vec<usize>,
    begin_index: collections::HashMap<u32, usize>,
}
impl WaveletMatrix {
    pub(crate) fn new(values: Vec<u32>) -> PyResult<Self> {
        let len = values.len();

        let height = bit_width(*values.par_iter().max().unwrap_or(&0u32));

        let mut zeros_count_per_layer = Vec::with_capacity(height);
        let mut layer_blocks_vec = Vec::with_capacity(height);
        let mut current_values = values;
        for depth in 0..height {
            let current_layer_bits = current_values
                .par_iter()
                .map(|value| (value >> (height - depth - 1) & 1u32).is_one())
                .collect::<Vec<_>>();
            let zeros_count = current_layer_bits.par_iter().filter(|&&b| !b).count();

            let mut reordered_values = vec![0u32; len];
            let mut zero_index = 0usize;
            let mut one_index = zeros_count;
            for (&bit, value) in iter::zip(&current_layer_bits, current_values) {
                if bit {
                    reordered_values[one_index] = value;
                    one_index += 1;
                } else {
                    reordered_values[zero_index] = value;
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
            current_values = reordered_values;
        }

        let mut value_begin_positions = collections::HashMap::new();
        current_values
            .into_iter()
            .enumerate()
            .for_each(|(position, value)| {
                value_begin_positions.entry(value).or_insert(position);
            });

        let layers = layer_blocks_vec
            .into_par_iter()
            .map(|blocks| BitVector::new(blocks, len))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(WaveletMatrix {
            len,
            height,
            layers,
            zeros_count_per_layer,
            begin_index: value_begin_positions,
        })
    }

    pub(crate) fn max_bit(&self) -> PyResult<usize> {
        Ok(self.height)
    }

    /// Get the value at the specified position.
    pub(crate) fn access(&self, mut index: usize) -> PyResult<u32> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        let mut result = 0u32;
        for (layer, zeros_count) in iter::zip(&self.layers, &self.zeros_count_per_layer) {
            let bit = layer.access(index)?;
            result <<= 1;
            if bit {
                result |= 1u32;
                index = zeros_count + layer.rank(bit, index)?;
            } else {
                index = layer.rank(bit, index)?;
            }
            debug_assert!(index <= self.len);
        }

        Ok(result)
    }

    /// Get all values in the Wavelet Matrix as a vector.
    pub(crate) fn values(&self) -> PyResult<Vec<u32>> {
        let mut indices = (0..self.len).collect::<Vec<usize>>();
        let mut values = vec![0u32; self.len];
        for (depth, (layer, zeros_count)) in
            iter::zip(&self.layers, &self.zeros_count_per_layer).enumerate()
        {
            let shift = self.height - depth - 1;
            let bits = layer
                .values()?
                .into_par_iter()
                .flat_map_iter(|block| {
                    (0..BlockType::BITS).map(move |i| ((block >> i) & BlockType::one()).is_one())
                })
                .collect::<Vec<_>>()
                .into_iter()
                .take(self.len)
                .collect::<Vec<_>>();
            let rank = iter::once([0usize; 2])
                .chain(bits.iter().scan([0usize; 2], |acc, &bit| {
                    acc[bit as usize] += 1;
                    Some(*acc)
                }))
                .collect::<Vec<_>>();
            indices
                .par_iter_mut()
                .zip(values.par_iter_mut())
                .for_each(|(index, value)| {
                    let bit = bits[*index];
                    if bit {
                        *value |= 1u32 << shift;
                        *index = zeros_count + rank[*index][bit as usize];
                    } else {
                        *index = rank[*index][bit as usize];
                    }
                    debug_assert!(*index <= self.len);
                });
        }
        Ok(values)
    }

    /// Count the number of occurrences of a value in the range [0, end).
    pub(crate) fn rank(&self, value: u32, mut end: usize) -> PyResult<usize> {
        if end > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        if bit_width(value) > self.height {
            return Ok(0usize);
        }

        let begin_index = match self.begin_index.get(&value) {
            Some(&index) => index,
            None => return Ok(0usize),
        };

        for (depth, (layer, zeros_count)) in
            iter::zip(&self.layers, &self.zeros_count_per_layer).enumerate()
        {
            let bit = (value >> (self.height - depth - 1) & 1u32).is_one();
            if bit {
                end = zeros_count + layer.rank(bit, end)?;
            } else {
                end = layer.rank(bit, end)?;
            }
            debug_assert!(end <= self.len);
        }

        debug_assert!(begin_index <= end);
        Ok(end - begin_index)
    }

    /// Find the position of the k-th occurrence of a value (1-indexed).
    pub(crate) fn select(&self, value: u32, kth: usize) -> PyResult<Option<usize>> {
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        if bit_width(value) > self.height {
            return Ok(None);
        }

        let begin_index = match self.begin_index.get(&value) {
            Some(&index) => index,
            None => return Ok(None),
        };

        let mut index = begin_index + kth - 1;
        for (depth, (layer, zeros_count)) in iter::zip(&self.layers, &self.zeros_count_per_layer)
            .enumerate()
            .rev()
        {
            let bit = (value >> (self.height - depth - 1) & 1u32).is_one();
            if bit {
                index -= zeros_count;
            }
            index = match layer.select(bit, index + 1)? {
                Some(index) => index,
                None => return Ok(None),
            };
            debug_assert!(index < self.len);
        }

        Ok(Some(index))
    }

    /// Get a list of values c in the range [start, end)
    pub(crate) fn range_list(
        &self,
        start: usize,
        end: usize,
    ) -> PyResult<collections::HashMap<u32, usize>> {
        if start > end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        struct StackItem {
            start: usize,
            end: usize,
            value: u32,
        }
        let mut stack = vec![StackItem {
            start,
            end,
            value: 0u32,
        }];

        for (layer, zeros_count) in iter::zip(&self.layers, &self.zeros_count_per_layer) {
            stack = stack.into_iter().try_fold(
                Vec::new(),
                |mut acc, item| -> PyResult<Vec<StackItem>> {
                    let StackItem { start, end, value } = item;

                    let start_zero = layer.rank(false, start)?;
                    let end_zero = layer.rank(false, end)?;

                    if start_zero != end_zero {
                        acc.push(StackItem {
                            start: start_zero,
                            end: end_zero,
                            value: value << 1,
                        });
                    }

                    let start_one = zeros_count + layer.rank(true, start)?;
                    let end_one = zeros_count + layer.rank(true, end)?;

                    if start_one != end_one {
                        acc.push(StackItem {
                            start: start_one,
                            end: end_one,
                            value: (value << 1) | 1u32,
                        });
                    }

                    Ok(acc)
                },
            )?;
        }

        let result = stack
            .into_iter()
            .filter(|StackItem { start, end, .. }| start != end)
            .map(|StackItem { start, end, value }| (value, end - start))
            .collect::<collections::HashMap<_, _>>();

        Ok(result)
    }

    // Count values in [start, end) with the top-k highest frequencies.
    pub(crate) fn topk(&self, start: usize, end: usize, k: usize) -> PyResult<Vec<(u32, usize)>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if k.is_zero() {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        let k = k.min(end - start);

        #[derive(cmp::PartialEq, Eq, PartialOrd, Ord)]
        struct QueueItem {
            len: usize,
            depth: usize,
            start: usize,
            end: usize,
            value: u32,
        }
        let mut heap = collections::BinaryHeap::from(vec![QueueItem {
            len: end - start,
            depth: 0,
            start,
            end,
            value: 0u32,
        }]);

        let mut result = Vec::with_capacity(k);
        while let Some(QueueItem {
            len,
            depth,
            start,
            end,
            value,
        }) = heap.pop()
        {
            if depth == self.height {
                result.push((value, len));
                if result.len() == k {
                    break;
                }
                continue;
            }

            let layer = &self.layers[depth];
            let zeros_count = self.zeros_count_per_layer[depth];

            let start_zero = layer.rank(false, start)?;
            let end_zero = layer.rank(false, end)?;
            debug_assert!(start_zero <= end_zero);

            let start_one = zeros_count + layer.rank(true, start)?;
            let end_one = zeros_count + layer.rank(true, end)?;
            debug_assert!(start_one <= end_one);

            if start_zero != end_zero {
                heap.push(QueueItem {
                    len: end_zero - start_zero,
                    depth: depth + 1,
                    start: start_zero,
                    end: end_zero,
                    value: value << 1usize,
                });
            }

            if end_one != start_one {
                heap.push(QueueItem {
                    len: end_one - start_one,
                    depth: depth + 1,
                    start: start_one,
                    end: end_one,
                    value: (value << 1usize) | 1u32,
                });
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn create_test_wavelet_matrix() -> WaveletMatrix {
        let test_data = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        WaveletMatrix::new(test_data).unwrap()
    }

    #[test]
    fn test_empty_wavelet_matrix() {
        Python::initialize();

        let wm = WaveletMatrix::new(vec![]).unwrap();

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

        // Range list should return empty map for empty range
        let result = wm.range_list(0, 0).unwrap();
        assert_eq!(result.len(), 0);

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
        assert_eq!(result.get(&0), Some(&64));
        assert_eq!(result.len(), 1);

        // Partial range
        let result = wm.range_list(10, 20).unwrap();
        assert_eq!(result.get(&0), Some(&10));
        assert_eq!(result.len(), 1);
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
        let result = wm.range_list(0, 64).unwrap();
        assert_eq!(result.get(&0), Some(&64));
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_maximum_value() {
        Python::initialize();

        let wm = WaveletMatrix::new(vec![u32::MAX; 64]).unwrap();

        assert_eq!(wm.access(1).unwrap(), u32::MAX);
        assert_eq!(wm.values().unwrap(), vec![u32::MAX; 64]);
        assert_eq!(wm.rank(u32::MAX, 1).unwrap(), 1);
        assert_eq!(wm.select(u32::MAX, 1).unwrap(), Some(0));

        // Range list should show all 64 values are u32::MAX
        let result = wm.range_list(0, 64).unwrap();
        assert_eq!(result.get(&u32::MAX), Some(&64));
        assert_eq!(result.len(), 1);

        // Partial range
        let result = wm.range_list(5, 15).unwrap();
        assert_eq!(result.get(&u32::MAX), Some(&10));
        assert_eq!(result.len(), 1);
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
        let result = wm.range_list(6, 8).unwrap();
        assert_eq!(result.get(&5), Some(&1));
        assert_eq!(result.get(&6), Some(&1));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_values_retrieval() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        let expected = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        assert_eq!(wm.values().unwrap(), expected);

        // Range list should match the count of values
        let result = wm.range_list(0, 12).unwrap();
        let expected_counts: collections::HashMap<u32, usize> =
            [(0, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 5), (6, 1)]
                .iter()
                .cloned()
                .collect();
        assert_eq!(result, expected_counts);
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
        let result = wm.range_list(0, 11).unwrap();
        assert_eq!(result.get(&5), Some(&5)); // 5 appears 5 times
        assert_eq!(result.get(&1), Some(&2)); // 1 appears 2 times
        assert_eq!(result.get(&0), None); // 0 is not in range
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
        let result = wm.range_list(6, 9).unwrap();
        assert_eq!(result.get(&5), Some(&1));
        assert_eq!(result.get(&6), Some(&1));
        assert_eq!(result.get(&1), Some(&1));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_range_list_full_range() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]

        let result = wm.range_list(0, 12).unwrap();

        assert_eq!(result.get(&0), Some(&1)); // 0 appears 1 time
        assert_eq!(result.get(&1), Some(&2)); // 1 appears 2 times
        assert_eq!(result.get(&2), Some(&1)); // 2 appears 1 time
        assert_eq!(result.get(&3), Some(&1)); // 3 appears 1 time
        assert_eq!(result.get(&4), Some(&1)); // 4 appears 1 time
        assert_eq!(result.get(&5), Some(&5)); // 5 appears 5 times
        assert_eq!(result.get(&6), Some(&1)); // 6 appears 1 time
        assert_eq!(result.len(), 7);
    }

    #[test]
    fn test_range_list_partial_range() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        // Indices:        0  1  2  3  4  5  6  7  8  9  10 11

        // Test range [0, 5): [5, 4, 5, 5, 2]
        let result = wm.range_list(0, 5).unwrap();
        assert_eq!(result.get(&2), Some(&1)); // 2 appears 1 time
        assert_eq!(result.get(&4), Some(&1)); // 4 appears 1 time
        assert_eq!(result.get(&5), Some(&3)); // 5 appears 3 times
        assert_eq!(result.len(), 3);

        // Test range [5, 10): [1, 5, 6, 1, 3]
        let result = wm.range_list(5, 10).unwrap();
        assert_eq!(result.get(&1), Some(&2)); // 1 appears 2 times
        assert_eq!(result.get(&3), Some(&1)); // 3 appears 1 time
        assert_eq!(result.get(&5), Some(&1)); // 5 appears 1 time
        assert_eq!(result.get(&6), Some(&1)); // 6 appears 1 time
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_range_list_empty_range() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        // Empty range: start == end
        let result = wm.range_list(5, 5).unwrap();
        assert_eq!(result.len(), 0);
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
        let result = wm.topk(0, 12, 3).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], (5, 5)); // 5 appears 5 times (most frequent)
        assert_eq!(result[1], (1, 2)); // 1 appears 2 times
        // Third element could be any of the values that appear once
        assert_eq!(result[2].1, 1);
    }

    #[test]
    fn test_topk_all_values() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]

        // Request more than unique values (should return all 7 unique values)
        let result = wm.topk(0, 12, 100).unwrap();
        assert_eq!(result.len(), 7);
        assert_eq!(result[0], (5, 5)); // 5 appears 5 times
        assert_eq!(result[1], (1, 2)); // 1 appears 2 times
        // Remaining 5 values appear once each
        assert!(result[2..].iter().all(|(_, count)| *count == 1));
    }

    #[test]
    fn test_topk_partial_range() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();
        // Test data: vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        // Indices:        0  1  2  3  4  5  6  7  8  9  10 11

        // Range [0, 5): [5, 4, 5, 5, 2]
        // Value frequencies: 5->3, 4->1, 2->1
        let result = wm.topk(0, 5, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (5, 3)); // 5 appears 3 times
        // Second element could be 4 or 2, both appear once
        assert_eq!(result[1].1, 1);

        // Range [5, 10): [1, 5, 6, 1, 3]
        // Value frequencies: 1->2, 5->1, 6->1, 3->1
        let result = wm.topk(5, 10, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, 2)); // 1 appears 2 times
        assert_eq!(result[1].1, 1); // One of 5, 6, or 3
    }

    #[test]
    fn test_topk_single_element() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        // Single element range
        let result = wm.topk(0, 1, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (5, 1)); // First element is 5
    }

    #[test]
    fn test_topk_all_same_frequency() {
        Python::initialize();

        // All different values with same frequency
        let wm = WaveletMatrix::new(vec![1, 2, 3, 4, 5]).unwrap();

        let result = wm.topk(0, 5, 3).unwrap();
        assert_eq!(result.len(), 3);
        // All have frequency 1
        assert!(result.iter().all(|(_, count)| *count == 1));
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

        let wm = WaveletMatrix::new(vec![]).unwrap();

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

        let result = wm.topk(0, 64, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (0, 64)); // All 64 values are 0

        // Partial range
        let result = wm.topk(10, 20, 5).unwrap();
        assert_eq!(result.len(), 1); // Only one unique value (0)
        assert_eq!(result[0], (0, 10));
    }

    #[test]
    fn test_topk_large_values() {
        Python::initialize();

        let wm = WaveletMatrix::new(vec![u32::MAX; 10]).unwrap();

        let result = wm.topk(0, 10, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (u32::MAX, 10));
    }
}
