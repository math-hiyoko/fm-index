use std::{cmp, collections, iter, ops};

use num_traits::{One, PrimInt, Unsigned, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyValueError},
};
use rayon::prelude::*;

use crate::utils::traits::{
    bit_vector::{BitVectorTrait, BlockType},
    bit_width::BitWidth,
};

/// A Wavelet Matrix data structure for efficient rank, select, and quantile queries.
///
/// The Wavelet Matrix decomposes a sequence into multiple bit vectors,
/// one for each bit position. This allows for efficient queries on the sequence.
pub(crate) trait WaveletMatrixTrait<NumberType, BitVectorType>
where
    Self: Sync,
    NumberType: PrimInt + Unsigned + BitWidth + Send + ops::BitOrAssign + ops::ShlAssign<usize>,
    BitVectorType: BitVectorTrait,
{
    /// Get the length of the Wavelet Matrix.
    fn len(&self) -> usize;

    /// Get the height (number of layers) of the Wavelet Matrix.
    fn max_bit(&self) -> usize;

    /// Get the bit vectors (layers) of the Wavelet Matrix.
    fn get_layers(&self) -> &[BitVectorType];

    /// Get the number of zeros in each layer.
    fn get_zeros_count_per_layer(&self) -> &[usize];

    /// Get the begin index for each unique value.
    #[inline]
    fn begin_index(&self, value: NumberType) -> Option<usize> {
        let mut start = 0usize;
        let mut end = self.len();
        for (depth, (layer, zeros_count)) in
            iter::zip(self.get_layers(), self.get_zeros_count_per_layer()).enumerate()
        {
            let bit = (value >> (self.max_bit() - depth - 1) & NumberType::one()).is_one();
            if bit {
                start = zeros_count + layer.rank(bit, start).ok()?;
                end = zeros_count + layer.rank(bit, end).ok()?;
            } else {
                start = layer.rank(bit, start).ok()?;
                end = layer.rank(bit, end).ok()?;
            }

            debug_assert!(end <= self.len());
            if start == end {
                break;
            }
        }

        debug_assert!(start <= end);
        if start == end { None } else { Some(start) }
    }

    /// Get all values in the Wavelet Matrix as a vector.
    fn values(&self) -> PyResult<Vec<NumberType>> {
        let mut indices = (0..self.len()).collect::<Vec<usize>>();
        let mut values = vec![NumberType::zero(); self.len()];
        for (depth, (layer, zeros_count)) in
            iter::zip(self.get_layers(), self.get_zeros_count_per_layer()).enumerate()
        {
            let bits = layer
                .values()?
                .into_par_iter()
                .flat_map_iter(|block| {
                    (0..BlockType::BITS).map(move |i| ((block >> i) & BlockType::one()).is_one())
                })
                .collect::<Vec<_>>()
                .into_iter()
                .take(self.len())
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
                        *value |= NumberType::one() << (self.max_bit() - depth - 1);
                        *index = zeros_count + rank[*index][bit as usize];
                    } else {
                        *index = rank[*index][bit as usize];
                    }
                    debug_assert!(*index <= self.len());
                });
        }
        Ok(values)
    }

    /// Get the value at the specified position.
    fn access(&self, mut index: usize) -> PyResult<NumberType> {
        if index >= self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        let mut result = NumberType::zero();
        for (layer, zeros_count) in iter::zip(self.get_layers(), self.get_zeros_count_per_layer()) {
            let bit = layer.access(index)?;
            result <<= 1usize;
            if bit {
                result |= NumberType::one();
                index = zeros_count + layer.rank(bit, index)?;
            } else {
                index = layer.rank(bit, index)?;
            }
            debug_assert!(index <= self.len());
        }

        Ok(result)
    }

    /// Count the number of occurrences of a value in the range [0, end).
    fn rank(&self, value: NumberType, mut end: usize) -> PyResult<usize> {
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if value.bit_width() > self.max_bit() {
            return Ok(0usize);
        }

        let begin_index = match self.begin_index(value) {
            Some(index) => index,
            None => return Ok(0usize),
        };

        for (depth, (layer, zeros_count)) in
            iter::zip(self.get_layers(), self.get_zeros_count_per_layer()).enumerate()
        {
            let bit = (value >> (self.max_bit() - depth - 1) & NumberType::one()).is_one();
            if bit {
                end = zeros_count + layer.rank(bit, end)?;
            } else {
                end = layer.rank(bit, end)?;
            }
            debug_assert!(end <= self.len());
        }

        debug_assert!(begin_index <= end);
        Ok(end - begin_index)
    }

    /// Find the position of the k-th occurrence of a value (1-indexed).
    fn select(&self, value: NumberType, kth: usize) -> PyResult<Option<usize>> {
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        if value.bit_width() > self.max_bit() {
            return Ok(None);
        }

        let begin_index = match self.begin_index(value) {
            Some(index) => index,
            None => return Ok(None),
        };

        let mut index = begin_index + kth - 1;
        for (depth, (layer, zeros_count)) in
            iter::zip(self.get_layers(), self.get_zeros_count_per_layer())
                .enumerate()
                .rev()
        {
            let bit = (value >> (self.max_bit() - depth - 1) & NumberType::one()).is_one();
            if bit {
                index -= zeros_count;
            }
            index = match layer.select(bit, index + 1)? {
                Some(index) => index,
                None => return Ok(None),
            };
            debug_assert!(index < self.len());
        }

        Ok(Some(index))
    }

    // Count values in [start, end) with the top-k highest frequencies.
    fn topk(&self, start: usize, end: usize, k: usize) -> PyResult<Vec<(NumberType, usize)>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if k.is_zero() {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }

        #[derive(cmp::PartialEq, Eq, PartialOrd, Ord)]
        struct QueueItem<T> {
            len: usize,
            depth: usize,
            start: usize,
            end: usize,
            value: T,
        }
        let mut heap = collections::BinaryHeap::new();
        heap.push(QueueItem::<NumberType> {
            len: end - start,
            depth: 0,
            start,
            end,
            value: NumberType::zero(),
        });

        let mut result = Vec::with_capacity(k);
        while let Some(QueueItem {
            len,
            depth,
            start,
            end,
            value,
        }) = heap.pop()
        {
            if depth == self.max_bit() {
                result.push((value, len));
                if result.len() == k {
                    break;
                }
                continue;
            }

            let layer = &self.get_layers()[depth];
            let zeros_count = self.get_zeros_count_per_layer()[depth];

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
                    value: (value << 1usize) | NumberType::one(),
                });
            }
        }

        Ok(result)
    }

    /// Get a list of values c in the range [start, end) such that lower <= c < upper.
    fn range_list(&self, start: usize, end: usize) -> PyResult<Vec<(NumberType, usize)>> {
        if start >= end {
            return Err(PyValueError::new_err("start must be less than end"));
        }
        if end > self.len() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }

        struct StackItem<T> {
            start: usize,
            end: usize,
            value: T,
        }
        let mut stack = vec![StackItem {
            start,
            end,
            value: NumberType::zero(),
        }];

        for (layer, zeros_count) in iter::zip(self.get_layers(), self.get_zeros_count_per_layer()) {
            stack = stack.into_iter().try_fold(
                Vec::new(),
                |mut acc, item| -> PyResult<Vec<StackItem<NumberType>>> {
                    let StackItem { start, end, value } = item;

                    let start_zero = layer.rank(false, start)?;
                    let end_zero = layer.rank(false, end)?;
                    debug_assert!(start_zero <= end_zero);

                    let start_one = zeros_count + layer.rank(true, start)?;
                    let end_one = zeros_count + layer.rank(true, end)?;
                    debug_assert!(start_one <= end_one);

                    let next_value_zero = value << 1;
                    if start_zero != end_zero {
                        acc.push(StackItem {
                            start: start_zero,
                            end: end_zero,
                            value: next_value_zero,
                        });
                    }

                    let next_value_one = (value << 1) | NumberType::one();
                    if start_one != end_one {
                        acc.push(StackItem {
                            start: start_one,
                            end: end_one,
                            value: next_value_one,
                        });
                    }

                    Ok(acc)
                },
            )?;
        }

        let result = stack
            .into_iter()
            .map(|StackItem { start, end, value }| (value, end - start))
            .collect::<Vec<_>>();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::marker;

    use pyo3::Python;

    use super::*;
    use crate::utils::traits::bit_vector::SampleBitVector;

    struct SampleWaveletMatrix<NumberType> {
        layers: Vec<SampleBitVector>,
        zeros_count_per_layer: Vec<usize>,
        height: usize,
        len: usize,
        phantom: marker::PhantomData<NumberType>,
    }

    impl<NumberType> SampleWaveletMatrix<NumberType>
    where
        NumberType: PrimInt + Unsigned + BitWidth,
    {
        fn new(data: &[NumberType]) -> Self {
            let mut values = data.to_vec();
            let height = values.iter().max().map_or(0usize, |max| max.bit_width());
            let len = values.len();
            let mut layers: Vec<SampleBitVector> = Vec::with_capacity(height);
            let mut zeros_count_per_layer: Vec<usize> = Vec::with_capacity(height);

            for i in 0..height {
                let mut bits = Vec::with_capacity(len);
                let mut zero_values = Vec::new();
                let mut one_values = Vec::new();
                for &value in values.iter() {
                    let bit = (value >> (height - i - 1) & NumberType::one()).is_one();
                    bits.push(bit);
                    if bit {
                        one_values.push(value);
                    } else {
                        zero_values.push(value);
                    }
                }
                layers.push(SampleBitVector::new(bits));
                zeros_count_per_layer.push(zero_values.len());
                values = [zero_values, one_values].concat();
            }

            SampleWaveletMatrix {
                layers,
                zeros_count_per_layer,
                height,
                len,
                phantom: marker::PhantomData,
            }
        }
    }

    impl<NumberType> WaveletMatrixTrait<NumberType, SampleBitVector> for SampleWaveletMatrix<NumberType>
    where
        NumberType:
            PrimInt + Unsigned + BitWidth + ops::ShlAssign<usize> + ops::BitOrAssign + Send + Sync,
    {
        fn get_layers(&self) -> &[SampleBitVector] {
            &self.layers
        }

        fn get_zeros_count_per_layer(&self) -> &[usize] {
            &self.zeros_count_per_layer
        }

        fn max_bit(&self) -> usize {
            self.height
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let wv = SampleWaveletMatrix::<u32>::new(&Vec::new());
        assert_eq!(wv.len(), 0);
        assert_eq!(wv.max_bit(), 0);
        assert_eq!(wv.values().unwrap(), Vec::<u32>::new());
        assert_eq!(
            wv.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv.rank(0, 0).unwrap(), 0);
        assert_eq!(wv.select(0, 1).unwrap(), None);
        assert_eq!(
            wv.topk(0, 0, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv.range_list(0, 0).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
    }

    #[test]
    fn test_all_zero() {
        Python::initialize();

        let wv = SampleWaveletMatrix::<u32>::new(&[0; 64]);
        assert_eq!(wv.len(), 64);
        assert_eq!(wv.max_bit(), 0);
        assert_eq!(wv.values().unwrap(), vec![0; 64]);
        assert_eq!(wv.access(1).unwrap(), 0);
        assert_eq!(wv.rank(0, 1).unwrap(), 1);
        assert_eq!(wv.select(0, 1).unwrap(), Some(0));
        assert_eq!(wv.topk(0, 64, 1).unwrap().len(), 1);
        assert_eq!(wv.range_list(0, 64).unwrap().len(), 1);
    }

    #[test]
    fn test_max_value() {
        Python::initialize();

        let wv = SampleWaveletMatrix::<u32>::new(&[u32::MAX - 1; 64]);
        assert_eq!(wv.len(), 64);
        assert_eq!(wv.max_bit(), 32);
        assert_eq!(wv.values().unwrap(), vec![u32::MAX - 1; 64]);
        assert_eq!(wv.access(1).unwrap(), u32::MAX - 1);
        assert_eq!(wv.rank(u32::MAX - 1, 1).unwrap(), 1);
        assert_eq!(wv.select(u32::MAX - 1, 1).unwrap(), Some(0));
        assert_eq!(wv.topk(0, 64, 1).unwrap().len(), 1);
        assert_eq!(wv.range_list(0, 64).unwrap().len(), 1);
    }

    #[test]
    fn test_methods() {
        Python::initialize();

        let elements: Vec<u32> = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];
        let wv = SampleWaveletMatrix::new(&elements);
        assert_eq!(
            wv.values().unwrap(),
            vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0],
        );
        assert_eq!(wv.access(0).unwrap(), 5);
        assert_eq!(wv.access(11).unwrap(), 0);
        assert_eq!(wv.rank(5, 12).unwrap(), 5);
        assert_eq!(wv.select(5, 3).unwrap(), Some(3));
        assert_eq!(wv.topk(0, 12, 3).unwrap(), vec![(5, 5), (1, 2), (3, 1)]);
        assert_eq!(
            wv.range_list(0, 12).unwrap(),
            vec![(0, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 5), (6, 1)],
        );
    }
}
