use std::{collections, hash, iter, ops};

use num_traits::{PrimInt, Unsigned, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyValueError},
};
use rayon::prelude::*;

use super::{bit_vector::BitVector, bit_width::BitWidth};

#[derive(Clone)]
pub(crate) struct WaveletMatrix<NumberType: PrimInt + Unsigned> {
    len: usize,
    is_none: BitVector,
    height: usize,
    layers: Vec<BitVector>,
    zeros: Vec<usize>,
    begin_index: collections::HashMap<NumberType, usize>,
}
impl<
    NumberType: hash::Hash + PrimInt + Unsigned + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync,
> WaveletMatrix<NumberType>
{
    pub(crate) fn new(data: Vec<Option<NumberType>>) -> PyResult<Self> {
        let len = data.len();

        let is_none = BitVector::new(data.iter().map(|value| value.is_none()).collect::<Vec<_>>())?;

        let mut current_values = data.into_iter().flatten().collect::<Vec<_>>();
        let height = current_values
            .par_iter()
            .max()
            .unwrap_or(&NumberType::zero())
            .bit_width();

        let mut zeros_count_per_layer = Vec::with_capacity(height);
        let mut layer_bits_vec = Vec::with_capacity(height);
        for depth in 0..height {
            let current_layer_bits = current_values
                .par_iter()
                .map(|&value| (value >> (height - depth - 1) & NumberType::one()).is_one())
                .collect::<Vec<_>>();
            let zeros_count = current_layer_bits.par_iter().filter(|&bit| !bit).count();

            let mut reordered_values = vec![NumberType::zero(); current_values.len()];
            let mut zeros_write_pos = 0usize;
            let mut ones_write_pos = zeros_count;
            for (&bit, value) in iter::zip(&current_layer_bits, current_values) {
                if bit {
                    reordered_values[ones_write_pos] = value;
                    ones_write_pos += 1;
                } else {
                    reordered_values[zeros_write_pos] = value;
                    zeros_write_pos += 1;
                }
            }

            zeros_count_per_layer.push(zeros_count);
            layer_bits_vec.push(current_layer_bits);
            current_values = reordered_values;
        }

        let layers = layer_bits_vec
            .into_par_iter()
            .map(BitVector::new)
            .collect::<PyResult<Vec<_>>>()?;

        let mut value_begin_positions = collections::HashMap::new();
        current_values
            .into_iter()
            .enumerate()
            .for_each(|(position, value)| {
                value_begin_positions.entry(value).or_insert(position);
            });

        Ok(WaveletMatrix {
            len,
            is_none,
            height,
            layers,
            zeros: zeros_count_per_layer,
            begin_index: value_begin_positions,
        })
    }

    pub(crate) fn max_bit(&self) -> PyResult<usize> {
        Ok(self.height)
    }

    /// Get the value at the specified position.
    pub(crate) fn access(&self, mut index: usize) -> PyResult<Option<NumberType>> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        if self.is_none.access(index)? {
            return Ok(None);
        }

        index -= self.is_none.rank(true, index)?;
        let mut reconstructed_value = NumberType::zero();
        for (layer, zeros_count) in iter::zip(&self.layers, &self.zeros) {
            let bit = layer.access(index)?;
            reconstructed_value <<= NumberType::one();
            if bit {
                reconstructed_value |= NumberType::one();
                index = zeros_count + layer.rank(bit, index)?;
            } else {
                index = layer.rank(bit, index)?;
            }
            debug_assert!(index <= layer.len());
        }

        Ok(Some(reconstructed_value))
    }

    /// Get all values in the Wavelet Matrix as a vector.
    pub(crate) fn values(&self) -> PyResult<Vec<Option<NumberType>>> {
        let non_none_count = self.is_none.rank(false, self.len)?;
        let mut non_none_indices = (0..non_none_count).collect::<Vec<_>>();
        let mut non_none_values = vec![NumberType::zero(); non_none_count];
        for (depth, (layer, zeros_count)) in iter::zip(&self.layers, &self.zeros).enumerate() {
            debug_assert_eq!(non_none_count, layer.len());
            let layer_bits = layer.values()?;
            let cumulative_rank = iter::once([0usize; 2])
                .chain(layer_bits.iter().scan([0usize; 2], |acc, &bit| {
                    acc[bit as usize] += 1;
                    Some(*acc)
                }))
                .collect::<Vec<_>>();
            non_none_indices
                .par_iter_mut()
                .zip(non_none_values.par_iter_mut())
                .for_each(|(index, value)| {
                    let bit = layer_bits[*index];
                    if bit {
                        *value |= NumberType::one() << (self.height - depth - 1);
                        *index = zeros_count + cumulative_rank[*index][bit as usize];
                    } else {
                        *index = cumulative_rank[*index][bit as usize];
                    }

                    debug_assert!(*index < layer.len());
                });
        }

        let none_flags = self.is_none.values()?;
        let mut result = Vec::with_capacity(self.len);
        let mut non_none_iter = non_none_values.iter();
        for &is_none_flag in none_flags.iter() {
            if is_none_flag {
                result.push(None);
            } else {
                result.push(non_none_iter.next().copied());
            }
        }

        Ok(result)
    }

    /// Count the number of occurrences of a value in the range [0, end).
    pub(crate) fn rank(&self, value: Option<NumberType>, mut end: usize) -> PyResult<usize> {
        if end > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        if value.is_none() {
            return self.is_none.rank(true, end);
        }
        end -= self.is_none.rank(true, end)?;

        let value = value.unwrap();
        if value.bit_width() > self.height {
            return Ok(0usize);
        }

        let value_start_pos = match self.begin_index.get(&value) {
            Some(&pos) => pos,
            None => return Ok(0usize),
        };

        for (depth, (layer, zeros_count)) in iter::zip(&self.layers, &self.zeros).enumerate() {
            let bit = (value >> (self.height - depth - 1) & NumberType::one()).is_one();
            if bit {
                end = zeros_count + layer.rank(bit, end)?;
            } else {
                end = layer.rank(bit, end)?;
            }
            debug_assert!(end <= layer.len());
        }

        debug_assert!(value_start_pos <= end);
        Ok(end - value_start_pos)
    }

    /// Find the position of the k-th occurrence of a value (1-indexed).
    pub(crate) fn select(&self, value: Option<NumberType>, kth: usize) -> PyResult<Option<usize>> {
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        if value.is_none() {
            return self.is_none.select(true, kth);
        }

        let value = value.unwrap();
        if value.bit_width() > self.height {
            return Ok(None);
        }

        let value_start_pos = match self.begin_index.get(&value) {
            Some(&pos) => pos,
            None => return Ok(None),
        };

        let mut index = value_start_pos + kth - 1;
        for (depth, (layer, zeros_count)) in iter::zip(&self.layers, &self.zeros).enumerate().rev()
        {
            let bit = (value >> (self.height - depth - 1) & NumberType::one()).is_one();
            if bit {
                index -= zeros_count;
            }
            index = match layer.select(bit, index + 1)? {
                Some(idx) => idx,
                None => return Ok(None),
            };
            debug_assert!(index < layer.len());
        }

        self.is_none.select(false, index + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn create_test_wavelet_matrix() -> WaveletMatrix<u8> {
        let test_data = vec![
            Some(5),
            Some(4),
            None,
            Some(5),
            Some(5),
            Some(2),
            None,
            Some(1),
            Some(5),
            Some(6),
            Some(1),
            None,
            Some(3),
            Some(5),
            Some(0),
        ];
        WaveletMatrix::new(test_data).unwrap()
    }

    #[test]
    fn test_empty_wavelet_matrix() {
        Python::initialize();

        let wm = WaveletMatrix::<u8>::new(vec![]).unwrap();

        // Access should fail on empty matrix
        assert_eq!(
            wm.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );

        // Values should return empty vector
        assert_eq!(wm.values().unwrap(), Vec::<Option<u8>>::new());

        // Rank should return 0
        assert_eq!(wm.rank(Some(0u8), 0).unwrap(), 0);
        assert_eq!(wm.rank(None, 0).unwrap(), 0);

        // Select with kth=0 should fail
        assert_eq!(
            wm.select(Some(0u8), 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
        assert_eq!(
            wm.select(None, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
    }

    #[test]
    fn test_all_zeros() {
        Python::initialize();

        let wm = WaveletMatrix::<u8>::new(vec![Some(0u8); 64]).unwrap();

        assert_eq!(wm.access(1).unwrap(), Some(0u8));
        assert_eq!(wm.values().unwrap(), vec![Some(0u8); 64]);
        assert_eq!(wm.rank(Some(0u8), 1).unwrap(), 1);
        assert_eq!(wm.select(Some(0u8), 1).unwrap(), Some(0));
    }

    #[test]
    fn test_all_none_values() {
        Python::initialize();

        let wm = WaveletMatrix::<u8>::new(vec![None; 64]).unwrap();

        assert_eq!(wm.access(1).unwrap(), None);
        assert_eq!(wm.values().unwrap(), vec![None; 64]);
        assert_eq!(wm.rank(None, 1).unwrap(), 1);
        assert_eq!(wm.select(None, 1).unwrap(), Some(0));
    }

    #[test]
    fn test_maximum_value() {
        Python::initialize();

        let wm = WaveletMatrix::<u8>::new(vec![Some(u8::MAX); 64]).unwrap();

        assert_eq!(wm.access(1).unwrap(), Some(u8::MAX));
        assert_eq!(wm.values().unwrap(), vec![Some(u8::MAX); 64]);
        assert_eq!(wm.rank(Some(u8::MAX), 1).unwrap(), 1);
        assert_eq!(wm.select(Some(u8::MAX), 1).unwrap(), Some(0));
    }

    #[test]
    fn test_access_operation() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        assert_eq!(wm.access(6).unwrap(), None);
        assert_eq!(wm.access(7).unwrap(), Some(1u8));
    }

    #[test]
    fn test_values_retrieval() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        let expected = vec![
            Some(5),
            Some(4),
            None,
            Some(5),
            Some(5),
            Some(2),
            None,
            Some(1),
            Some(5),
            Some(6),
            Some(1),
            None,
            Some(3),
            Some(5),
            Some(0),
        ];
        assert_eq!(wm.values().unwrap(), expected);
    }

    #[test]
    fn test_rank_operation() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        assert_eq!(wm.rank(Some(5u8), 11).unwrap(), 4);
        assert_eq!(wm.rank(None, 11).unwrap(), 2);
    }

    #[test]
    fn test_select_operation() {
        Python::initialize();

        let wm = create_test_wavelet_matrix();

        // Valid selections
        assert_eq!(wm.select(Some(5u8), 4).unwrap(), Some(8));
        assert_eq!(wm.select(None, 3).unwrap(), Some(11));

        // Out of range selections
        assert_eq!(wm.select(Some(5u8), 6).unwrap(), None);
        assert_eq!(wm.select(None, 6).unwrap(), None);
    }
}
