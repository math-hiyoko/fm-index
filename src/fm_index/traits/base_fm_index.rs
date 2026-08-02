use std::collections;

use num_traits::Zero;
use pyo3::PyResult;
use rayon::prelude::*;

use crate::utils::traits::{bit_vector::BitVectorTrait, wavelet_matrix::WaveletMatrixTrait};

pub(crate) const ARRAY_SAMPLING_RATE: usize = 1 << 4;

pub(crate) trait BaseFMIndexTrait: Sync {
    type BitVector: BitVectorTrait;
    type WaveletMatrix: WaveletMatrixTrait<u32, Self::BitVector>;

    fn len(&self) -> usize;

    fn get_zero_suffix_idx(&self) -> usize;

    fn get_suffix_idx_sampled(&self) -> &[usize];

    fn get_counts_less(&self) -> &collections::HashMap<u32, usize>;

    fn get_burrows_wheeler_transform(&self) -> &Self::WaveletMatrix;

    #[inline]
    fn lf_mapping(&self, index: usize) -> PyResult<usize> {
        let symbol = self.get_burrows_wheeler_transform().access(index)?;
        if symbol.is_zero() && index == self.get_zero_suffix_idx() {
            return Ok(0);
        }

        let rank = self.get_burrows_wheeler_transform().rank(symbol, index)?;
        if symbol.is_zero() {
            if index < self.get_zero_suffix_idx() {
                return Ok(rank + 1);
            } else {
                return Ok(rank);
            }
        }

        let counts_less = self.get_counts_less();
        let count_less = counts_less[&symbol];
        Ok(count_less + rank)
    }

    #[inline]
    fn suffix_idx(&self, mut index: usize) -> PyResult<usize> {
        let mut steps = 0usize;
        while !index.is_multiple_of(ARRAY_SAMPLING_RATE) {
            index = self.lf_mapping(index)?;
            steps += 1;
        }
        let mut idx = self.get_suffix_idx_sampled()[index / ARRAY_SAMPLING_RATE] + steps;
        if idx >= self.len() {
            idx -= self.len();
        }
        Ok(idx)
    }

    #[inline]
    fn values(&self) -> PyResult<Vec<u32>> {
        let mut values = vec![0u32; self.len()];

        if self.len() > 0 {
            let suffix_idx_0 = self.get_suffix_idx_sampled()[0];
            let mut index = if suffix_idx_0.is_zero() {
                self.len() - 1
            } else {
                suffix_idx_0 - 1
            };
            let mut value_idx = 0usize;
            let lf_mapping = (0..self.len())
                .into_par_iter()
                .map(|index| self.lf_mapping(index))
                .collect::<PyResult<Vec<_>>>()?;
            let bwt_values = self.get_burrows_wheeler_transform().values()?;
            for _ in 0..self.len() {
                values[index] = bwt_values[value_idx];
                index = if index.is_zero() {
                    self.len() - 1
                } else {
                    index - 1
                };
                value_idx = lf_mapping[value_idx];
            }
        }

        Ok(values)
    }

    fn range_search(&self, pattern: Vec<u32>) -> PyResult<(usize, usize)> {
        let (mut start, mut end) = (0usize, self.len());
        for symbol in pattern.into_iter().rev() {
            let count_less = match self.get_counts_less().get(&symbol) {
                Some(&count) => count,
                None => return Ok((0, 0)),
            };
            start = count_less + self.get_burrows_wheeler_transform().rank(symbol, start)?;
            end = count_less + self.get_burrows_wheeler_transform().rank(symbol, end)?;

            debug_assert!(start <= end && end <= self.len());
            if start == end {
                break;
            }
        }

        Ok((start, end))
    }
}
