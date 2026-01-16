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

        let is_none =
            BitVector::new(&data.iter().map(|value| value.is_none()).collect::<Vec<_>>())?;

        let mut values_some = data.into_iter().flatten().collect::<Vec<_>>();
        let height = values_some
            .par_iter()
            .max()
            .unwrap_or(&NumberType::zero())
            .bit_width();

        let mut zeros = Vec::with_capacity(height);
        let mut bits = Vec::with_capacity(height);
        for i in 0..height {
            let layer_bits = values_some
                .par_iter()
                .map(|&value| (value >> (height - i - 1) & NumberType::one()).is_one())
                .collect::<Vec<_>>();
            let num_zeros = layer_bits.par_iter().filter(|&bit| !bit).count();

            let mut next_values = vec![NumberType::zero(); values_some.len()];
            let mut zero_index = 0usize;
            let mut one_index = num_zeros;
            for (&bit, value) in iter::zip(&layer_bits, values_some) {
                if bit {
                    next_values[one_index] = value;
                    one_index += 1;
                } else {
                    next_values[zero_index] = value;
                    zero_index += 1;
                }
            }

            zeros.push(num_zeros);
            bits.push(layer_bits);
            values_some = next_values;
        }

        let layers = bits
            .into_par_iter()
            .map(|layer_bits| BitVector::new(&layer_bits))
            .collect::<PyResult<Vec<_>>>()?;

        let mut begin_index = collections::HashMap::new();
        values_some
            .into_iter()
            .enumerate()
            .for_each(|(index, value)| {
                begin_index.entry(value).or_insert(index);
            });

        Ok(WaveletMatrix {
            len,
            is_none,
            height,
            layers,
            zeros,
            begin_index,
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
        let mut result = NumberType::zero();
        for (layer, zero) in iter::zip(&self.layers, &self.zeros) {
            let bit = layer.access(index)?;
            result <<= NumberType::one();
            if bit {
                result |= NumberType::one();
                index = zero + layer.rank(bit, index)?;
            } else {
                index = layer.rank(bit, index)?;
            }
            debug_assert!(index <= layer.len());
        }

        Ok(Some(result))
    }

    /// Get all values in the Wavelet Matrix as a vector.
    pub(crate) fn values(&self) -> PyResult<Vec<Option<NumberType>>> {
        let num_some = self.is_none.rank(false, self.len)?;
        let mut indices_some = (0..num_some).collect::<Vec<_>>();
        let mut values_some = vec![NumberType::zero(); num_some];
        for (depth, (layer, zero)) in iter::zip(&self.layers, &self.zeros).enumerate() {
            debug_assert_eq!(num_some, layer.len());
            let bits = layer.values()?;
            let rank = iter::once([0usize; 2])
                .chain(bits.iter().scan([0usize; 2], |acc, &bit| {
                    acc[bit as usize] += 1;
                    Some(*acc)
                }))
                .collect::<Vec<_>>();
            indices_some
                .par_iter_mut()
                .zip(values_some.par_iter_mut())
                .for_each(|(index, value)| {
                    let bit = bits[*index];
                    if bit {
                        *value |= NumberType::one() << (self.height - depth - 1);
                        *index = zero + rank[*index][bit as usize];
                    } else {
                        *index = rank[*index][bit as usize];
                    }

                    debug_assert!(*index < layer.len());
                });
        }

        let is_none = self.is_none.values()?;
        let mut values = Vec::with_capacity(self.len);
        let mut iter_some = values_some.iter();
        for &is_none in is_none.iter() {
            if is_none {
                values.push(None);
            } else {
                values.push(iter_some.next().copied());
            }
        }

        Ok(values)
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

        let begin_index = match self.begin_index.get(&value) {
            Some(&index) => index,
            None => return Ok(0usize),
        };

        for (depth, (layer, zero)) in iter::zip(&self.layers, &self.zeros).enumerate() {
            let bit = (value >> (self.height - depth - 1) & NumberType::one()).is_one();
            if bit {
                end = zero + layer.rank(bit, end)?;
            } else {
                end = layer.rank(bit, end)?;
            }
            debug_assert!(end <= layer.len());
        }

        debug_assert!(begin_index <= end);
        Ok(end - begin_index)
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

        let begin_index = match self.begin_index.get(&value) {
            Some(&index) => index,
            None => return Ok(None),
        };

        let mut index = begin_index + kth - 1;
        for (depth, (layer, zero)) in iter::zip(&self.layers, &self.zeros).enumerate().rev() {
            let bit = (value >> (self.height - depth - 1) & NumberType::one()).is_one();
            if bit {
                index -= zero;
            }
            index = match layer.select(bit, index + 1)? {
                Some(index) => index,
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

    fn create_dummy() -> WaveletMatrix<u8> {
        let elements = [
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
        ]
        .to_vec();
        WaveletMatrix::new(elements).unwrap()
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let wv = WaveletMatrix::<u8>::new(Vec::new()).unwrap();
        assert_eq!(
            wv.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv.values().unwrap(), Vec::<Option<u8>>::new());
        assert_eq!(wv.rank(Some(0u8), 0).unwrap(), 0);
        assert_eq!(wv.rank(None, 0).unwrap(), 0);
        assert_eq!(
            wv.select(Some(0u8), 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
        assert_eq!(
            wv.select(None, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
    }

    #[test]
    fn test_all_zero() {
        Python::initialize();

        let wv = WaveletMatrix::<u8>::new([Some(0u8); 64].to_vec()).unwrap();
        assert_eq!(wv.access(1).unwrap(), Some(0u8));
        assert_eq!(wv.values().unwrap(), [Some(0u8); 64]);
        assert_eq!(wv.rank(Some(0u8), 1).unwrap(), 1);
        assert_eq!(wv.select(Some(0u8), 1).unwrap(), Some(0));
    }

    #[test]
    fn test_all_none() {
        Python::initialize();

        let wv = WaveletMatrix::<u8>::new([None; 64].to_vec()).unwrap();
        assert_eq!(wv.access(1).unwrap(), None);
        assert_eq!(wv.values().unwrap(), [None; 64]);
        assert_eq!(wv.rank(None, 1).unwrap(), 1);
        assert_eq!(wv.select(None, 1).unwrap(), Some(0));
    }

    #[test]
    fn test_max_value() {
        Python::initialize();

        let wv = WaveletMatrix::<u8>::new([Some(u8::MAX); 64].to_vec()).unwrap();
        assert_eq!(wv.access(1).unwrap(), Some(u8::MAX));
        assert_eq!(wv.values().unwrap(), [Some(u8::MAX); 64]);
        assert_eq!(wv.rank(Some(u8::MAX), 1).unwrap(), 1);
        assert_eq!(wv.select(Some(u8::MAX), 1).unwrap(), Some(0));
    }

    #[test]
    fn test_access() {
        Python::initialize();

        let wv = create_dummy();
        assert_eq!(wv.access(6).unwrap(), None);
        assert_eq!(wv.access(7).unwrap(), Some(1u8));
    }

    #[test]
    fn test_values() {
        Python::initialize();

        let wv = create_dummy();
        assert_eq!(
            wv.values().unwrap(),
            [
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
                Some(0)
            ],
        );
    }

    #[test]
    fn test_rank() {
        Python::initialize();

        let wv = create_dummy();
        assert_eq!(wv.rank(Some(5u8), 11).unwrap(), 4usize);
        assert_eq!(wv.rank(None, 11).unwrap(), 2usize);
    }

    #[test]
    fn test_select() {
        Python::initialize();

        let wv = create_dummy();
        assert_eq!(wv.select(Some(5u8), 4).unwrap(), Some(8usize));
        assert_eq!(wv.select(None, 3).unwrap(), Some(11usize));
        assert_eq!(wv.select(Some(5u8), 6).unwrap(), None);
        assert_eq!(wv.select(None, 6).unwrap(), None);
    }
}
