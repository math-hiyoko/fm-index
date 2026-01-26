use std::collections;

use num_traits::Zero;
use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::utils::wavelet_matrix::WaveletMatrix;

const ARRAY_SAMPLING_RATE: usize = 1 << 4;

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct BaseFMIndex {
    len: usize,
    zero_suffix_idx: usize,
    suffix_idx_sampled: Vec<usize>,
    counts_less: collections::HashMap<u32, usize>,
    burrows_wheeler_transform: WaveletMatrix,
}

impl BaseFMIndex {
    pub(super) fn new(data: Vec<u32>, suffix_idx: Vec<usize>) -> PyResult<Self> {
        let len = data.len();

        let zero_suffix_idx = suffix_idx
            .par_iter()
            .position_any(|&idx| idx == 0)
            .unwrap_or(0usize);
        let suffix_idx_sampled = suffix_idx
            .iter()
            .step_by(ARRAY_SAMPLING_RATE)
            .copied()
            .collect::<Vec<_>>();

        let mut counts_less = collections::HashMap::new();
        for (cumulative_count, &idx) in suffix_idx.iter().enumerate() {
            let symbol = data[idx];
            counts_less.entry(symbol).or_insert(cumulative_count);
        }

        let burrows_wheeler_transform = suffix_idx
            .into_par_iter()
            .map(|idx| {
                if idx == 0 {
                    data[len - 1]
                } else {
                    data[idx - 1]
                }
            })
            .collect::<Vec<_>>();
        let burrows_wheeler_transform = WaveletMatrix::new(burrows_wheeler_transform)?;

        Ok(BaseFMIndex {
            len,
            zero_suffix_idx,
            suffix_idx_sampled,
            counts_less,
            burrows_wheeler_transform,
        })
    }

    #[inline]
    pub(super) fn lf_mapping(&self, index: usize) -> PyResult<usize> {
        let bwt = &self.burrows_wheeler_transform;
        let symbol = bwt.access(index)?;
        if symbol.is_zero() && index == self.zero_suffix_idx {
            return Ok(0);
        }

        let rank = bwt.rank(symbol, index)?;
        if symbol.is_zero() {
            if index < self.zero_suffix_idx {
                return Ok(rank + 1);
            } else {
                return Ok(rank);
            }
        }

        let count_less = self.counts_less[&symbol];
        Ok(count_less + rank)
    }

    #[inline]
    pub(super) fn suffix_idx(&self, mut index: usize) -> PyResult<usize> {
        let mut steps = 0usize;
        while !index.is_multiple_of(ARRAY_SAMPLING_RATE) {
            index = self.lf_mapping(index)?;
            steps += 1;
        }
        let suffix_idx_sampled = self.suffix_idx_sampled[index / ARRAY_SAMPLING_RATE];
        let mut idx = suffix_idx_sampled + steps;
        if idx >= self.len {
            idx -= self.len;
        }
        Ok(idx)
    }

    #[inline]
    pub(super) fn zero_suffix_idx(&self) -> usize {
        self.zero_suffix_idx
    }

    #[inline]
    pub(super) fn burrows_wheeler_transform(&self) -> &WaveletMatrix {
        &self.burrows_wheeler_transform
    }

    #[inline]
    pub(super) fn values(&self) -> PyResult<Vec<u32>> {
        let mut values = vec![0u32; self.len];

        if self.len > 0 {
            let mut index = if self.suffix_idx_sampled[0].is_zero() {
                self.len - 1
            } else {
                self.suffix_idx_sampled[0] - 1
            };
            let mut value_idx = 0usize;
            let lf_mapping = (0..self.len)
                .into_par_iter()
                .map(|index| self.lf_mapping(index))
                .collect::<PyResult<Vec<_>>>()?;
            let bwt_values = self.burrows_wheeler_transform.values()?;
            for _ in 0..self.len {
                values[index] = bwt_values[value_idx];
                index = if index.is_zero() {
                    self.len - 1
                } else {
                    index - 1
                };
                value_idx = lf_mapping[value_idx];
            }
        }

        Ok(values)
    }

    #[inline]
    pub(super) fn range_search(&self, pattern: Vec<u32>) -> PyResult<(usize, usize)> {
        let (mut start, mut end) = (0usize, self.len);
        for symbol in pattern.into_iter().rev() {
            let count_less = match self.counts_less.get(&symbol) {
                Some(&count) => count,
                None => return Ok((0, 0)),
            };
            start = count_less + self.burrows_wheeler_transform.rank(symbol, start)?;
            end = count_less + self.burrows_wheeler_transform.rank(symbol, end)?;

            debug_assert!(start <= end && end <= self.len);
            if start == end {
                break;
            }
        }

        Ok((start, end))
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::utils::suffix_array::suffix_array;

    #[test]
    fn test_base_fm_index_empty() {
        let data = vec![0];
        let suffix_idx = suffix_array(&data);
        let fm_index = BaseFMIndex::new(data, suffix_idx).unwrap();

        assert_eq!(fm_index.suffix_idx(0).unwrap(), 0);
        assert_eq!(fm_index.values().unwrap(), vec![0]);
        assert_eq!(fm_index.range_search(vec![0]).unwrap(), (0, 1));
    }

    #[test]
    fn test_base_fm_index_single_char() {
        let data = "aaaaaaaaaa"
            .chars()
            .map(|c| c as u32)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let suffix_idx = suffix_array(&data);
        let fm_index = BaseFMIndex::new(data.to_vec(), suffix_idx.clone()).unwrap();

        for (i, &suffix_idx) in suffix_idx.iter().enumerate().take(data.len()) {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(
            fm_index
                .range_search(vec![b'a' as u32, b'a' as u32, b'a' as u32])
                .unwrap(),
            (3, 11),
        );
    }

    #[test]
    fn test_base_fm_index_u32() {
        let data = "にわにはにわにわとりがいる"
            .chars()
            .map(|c| c as u32)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let suffix_idx = suffix_array(&data);
        let fm_index = BaseFMIndex::new(data.clone(), suffix_idx.clone()).unwrap();

        for (i, &suffix_idx) in suffix_idx.iter().enumerate() {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(
            fm_index
                .range_search(vec!['に' as u32, 'わ' as u32])
                .unwrap(),
            (5, 8),
        );
    }
}
