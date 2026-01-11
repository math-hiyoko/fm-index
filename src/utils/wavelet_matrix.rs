use std::{collections, iter};

use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyValueError},
};

use super::{bit_vector::BitVector, bit_width::bit_width};

#[derive(Clone)]
pub(crate) struct WaveletMatrix {
    len: usize,
    is_none: BitVector,
    height: usize,
    layers: Vec<BitVector>,
    zeros: Vec<usize>,
    begin_index: collections::HashMap<u8, usize>,
}
impl WaveletMatrix {
    pub(crate) fn new(data: &[Option<u8>]) -> PyResult<Self> {
        let len = data.len();

        let values = data.to_owned();
        let is_none = BitVector::new(
            &values
                .iter()
                .map(|value| value.is_none())
                .collect::<Vec<_>>(),
        )?;

        let mut values_some = values.iter().filter_map(|&value| value).collect::<Vec<_>>();
        let height = bit_width(values_some.iter().max().unwrap_or(&u8::zero()));
        let mut layers: Vec<BitVector> = Vec::with_capacity(height);
        let mut zeros: Vec<usize> = Vec::with_capacity(height);

        for i in 0..height {
            let bits = values_some
                .iter()
                .map(|&value| (value >> (height - i - 1) & u8::one()).is_one())
                .collect::<Vec<_>>();
            let num_zeros = bits.iter().filter(|&&bit| !bit).count();
            layers.push(BitVector::new(&bits)?);
            zeros.push(num_zeros);

            let mut next_values = vec![u8::zero(); values_some.len()];
            let mut zero_index = 0usize;
            let mut one_index = num_zeros;
            for (bit, value) in iter::zip(bits, values_some) {
                if bit {
                    next_values[one_index] = value;
                    one_index += 1;
                } else {
                    next_values[zero_index] = value;
                    zero_index += 1;
                }
            }
            values_some = next_values;
        }

        let mut begin_index = collections::HashMap::new();
        values_some.iter().enumerate().for_each(|(i, &v)| {
            begin_index.entry(v).or_insert(i);
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

    /// Get the value at the specified position.
    pub(crate) fn access(&self, mut index: usize) -> PyResult<Option<u8>> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        if self.is_none.access(index)? {
            return Ok(None);
        }

        index -= self.is_none.rank(true, index)?;
        let mut result = u8::zero();
        for (layer, zero) in iter::zip(&self.layers, &self.zeros) {
            let bit = layer.access(index)?;
            result <<= u8::one();
            if bit {
                result |= u8::one();
                index = zero + layer.rank(bit, index)?;
            } else {
                index = layer.rank(bit, index)?;
            }
            debug_assert!(index <= layer.len());
        }

        Ok(Some(result))
    }

    /// Get all values in the Wavelet Matrix as a vector.
    pub(crate) fn values(&self) -> PyResult<Vec<Option<u8>>> {
        let num_some = self.is_none.rank(false, self.len)?;
        let mut indices_some = (0..num_some).collect::<Vec<_>>();
        let mut values_some = vec![u8::zero(); num_some];
        for (depth, (layer, zero)) in iter::zip(&self.layers, &self.zeros).enumerate() {
            debug_assert_eq!(num_some, layer.len());
            let bits = layer.values()?;
            let rank = iter::once([0usize; 2])
                .chain(bits.iter().scan([0usize; 2], |acc, &bit| {
                    acc[bit as usize] += 1;
                    Some(*acc)
                }))
                .collect::<Vec<_>>();
            for (index, value) in iter::zip(indices_some.iter_mut(), values_some.iter_mut()) {
                let bit = bits[*index];
                if bit {
                    *value |= u8::one() << (self.height - depth - 1);
                    *index = zero + rank[*index][bit as usize];
                } else {
                    *index = rank[*index][bit as usize];
                }
                debug_assert!(*index < layer.len());
            }
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
    pub(crate) fn rank(&self, value: &Option<u8>, mut end: usize) -> PyResult<usize> {
        if end > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        if value.is_none() {
            return self.is_none.rank(true, end);
        }
        end -= self.is_none.rank(true, end)?;

        let value = value.as_ref().unwrap();
        if bit_width(value) > self.height {
            return Ok(0usize);
        }

        let begin_index = match self.begin_index.get(value) {
            Some(&index) => index,
            None => return Ok(0usize),
        };

        for (depth, (layer, zero)) in iter::zip(&self.layers, &self.zeros).enumerate() {
            let bit = (value >> (self.height - depth - 1) & u8::one()).is_one();
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
    pub(crate) fn select(&self, value: &Option<u8>, kth: usize) -> PyResult<Option<usize>> {
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        if value.is_none() {
            return self.is_none.select(true, kth);
        }

        let value = value.as_ref().unwrap();
        if bit_width(value) > self.height {
            return Ok(None);
        }

        let begin_index = match self.begin_index.get(value) {
            Some(&index) => index,
            None => return Ok(None),
        };

        let mut index = begin_index + kth - 1;
        for (depth, (layer, zero)) in iter::zip(&self.layers, &self.zeros).enumerate().rev() {
            let bit = (value >> (self.height - depth - 1) & u8::one()).is_one();
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

    fn create_dummy() -> WaveletMatrix {
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
        ];
        WaveletMatrix::new(&elements).unwrap()
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let wv = WaveletMatrix::new(&Vec::new()).unwrap();
        assert_eq!(
            wv.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv.values().unwrap(), Vec::<Option<u8>>::new());
        assert_eq!(wv.rank(&Some(0u8), 0).unwrap(), 0);
        assert_eq!(wv.rank(&None, 0).unwrap(), 0);
        assert_eq!(
            wv.select(&Some(0u8), 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
        assert_eq!(
            wv.select(&None, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
    }

    #[test]
    fn test_all_zero() {
        Python::initialize();

        let wv = WaveletMatrix::new(&[Some(0u8); 64]).unwrap();
        assert_eq!(wv.access(1).unwrap(), Some(0u8));
        assert_eq!(wv.values().unwrap(), [Some(0u8); 64]);
        assert_eq!(wv.rank(&Some(0u8), 1).unwrap(), 1);
        assert_eq!(wv.select(&Some(0u8), 1).unwrap(), Some(0));
    }

    #[test]
    fn test_all_none() {
        Python::initialize();

        let wv = WaveletMatrix::new(&[None; 64]).unwrap();
        assert_eq!(wv.access(1).unwrap(), None);
        assert_eq!(wv.values().unwrap(), [None; 64]);
        assert_eq!(wv.rank(&None, 1).unwrap(), 1);
        assert_eq!(wv.select(&None, 1).unwrap(), Some(0));
    }

    #[test]
    fn test_max_value() {
        Python::initialize();

        let wv = WaveletMatrix::new(&[Some(u8::MAX); 64]).unwrap();
        assert_eq!(wv.access(1).unwrap(), Some(u8::MAX));
        assert_eq!(wv.values().unwrap(), [Some(u8::MAX); 64]);
        assert_eq!(wv.rank(&Some(u8::MAX), 1).unwrap(), 1);
        assert_eq!(wv.select(&Some(u8::MAX), 1).unwrap(), Some(0));
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
        assert_eq!(wv.rank(&Some(5u8), 11).unwrap(), 4usize);
        assert_eq!(wv.rank(&None, 11).unwrap(), 2usize);
    }

    #[test]
    fn test_select() {
        Python::initialize();

        let wv = create_dummy();
        assert_eq!(wv.select(&Some(5u8), 4).unwrap(), Some(8usize));
        assert_eq!(wv.select(&None, 3).unwrap(), Some(11usize));
        assert_eq!(wv.select(&Some(5u8), 6).unwrap(), None);
        assert_eq!(wv.select(&None, 6).unwrap(), None);
    }
}
