use std::{collections, hash, ops};

use num_traits::{PrimInt, Unsigned, Zero};
use pyo3::PyResult;
use rayon::prelude::*;

use crate::utils::{
    bit_width::BitWidth, suffix_array::suffix_array_option, wavelet_matrix::WaveletMatrix,
};

pub(super) const SUFFIX_ARRAY_SAMPLING_RATE: usize = 32;

#[derive(Clone)]
pub(super) struct BaseFMIndex<
    Element: PrimInt + Unsigned + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync,
> {
    len: usize,
    zero_suffix_idx: usize,
    suffix_idx_sampled: Vec<usize>,
    counts_less: collections::HashMap<Option<Element>, usize>,
    burrows_wheeler_transform: WaveletMatrix<Element>,
}

impl<
    Element: PrimInt + Unsigned + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync,
> BaseFMIndex<Element>
{
    pub(super) fn new(data: Vec<Option<Element>>) -> PyResult<Self> {
        let suffix_idx = suffix_array_option(data.clone());

        Self::new_with_suffix_array(data, suffix_idx)
    }

    pub(super) fn new_with_suffix_array(
        data: Vec<Option<Element>>,
        suffix_idx: Vec<usize>,
    ) -> PyResult<Self> {
        let len = data.len();

        let zero_suffix_idx = suffix_idx
            .par_iter()
            .position_any(|&idx| idx == 0)
            .unwrap_or(0usize);
        let suffix_idx_sampled = suffix_idx
            .iter()
            .step_by(SUFFIX_ARRAY_SAMPLING_RATE)
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
        if symbol.is_none() && index == self.zero_suffix_idx {
            return Ok(0);
        }

        let rank = bwt.rank(symbol, index)?;
        if symbol.is_none() {
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
        while !index.is_multiple_of(SUFFIX_ARRAY_SAMPLING_RATE) {
            index = self.lf_mapping(index)?;
            steps += 1;
        }
        let suffix_idx_sampled = self.suffix_idx_sampled[index / SUFFIX_ARRAY_SAMPLING_RATE];
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
    pub(super) fn burrows_wheeler_transform(&self) -> &WaveletMatrix<Element> {
        &self.burrows_wheeler_transform
    }

    #[inline]
    pub(super) fn values(&self) -> PyResult<Vec<Option<Element>>> {
        let mut values = vec![None; self.len];

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
    pub(super) fn range_search(&self, pattern: Vec<Option<Element>>) -> PyResult<(usize, usize)> {
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

    #[test]
    fn test_base_fm_index_empty() {
        let data = [Option::<u8>::None];
        let fm_index = BaseFMIndex::new(data.to_vec()).unwrap();

        assert_eq!(fm_index.suffix_idx(0).unwrap(), 0);
        assert_eq!(fm_index.values().unwrap(), [None]);
        assert_eq!(fm_index.range_search([None].to_vec()).unwrap(), (0, 1));
    }

    #[test]
    fn test_base_fm_index_single_char() {
        let data = b"aaaaaaaaaa"
            .to_vec()
            .into_iter()
            .map(Some)
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let fm_index = BaseFMIndex::new(data.to_vec()).unwrap();
        let suffix_idx = suffix_array_option(data.clone());

        for (i, &suffix_idx) in suffix_idx.iter().enumerate().take(data.len()) {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(
            fm_index
                .range_search([Some(b'a'), Some(b'a'), Some(b'a')].to_vec())
                .unwrap(),
            (3, 11),
        );
    }

    #[test]
    fn test_base_fm_index_u8() {
        let data = b"mississippi"
            .to_vec()
            .into_iter()
            .map(Some)
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let fm_index = BaseFMIndex::new(data.to_vec()).unwrap();
        let suffix_idx = suffix_array_option(data.clone());

        for (i, &suffix_idx) in suffix_idx.iter().enumerate().take(data.len()) {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(
            fm_index
                .range_search([Some(b's'), Some(b'i')].to_vec())
                .unwrap(),
            (8, 10),
        );
    }

    #[test]
    fn test_base_fm_index_u32() {
        let data = "にわにはにわにわとりがいる"
            .chars()
            .map(|c| Some(c as u32))
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let fm_index = BaseFMIndex::new(data.to_vec()).unwrap();
        let suffix_idx = suffix_array_option(data.clone());

        for (i, &suffix_idx) in suffix_idx.iter().enumerate().take(data.len()) {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(
            fm_index
                .range_search([Some('に' as u32), Some('わ' as u32)].to_vec())
                .unwrap(),
            (5, 8),
        );
    }
}
