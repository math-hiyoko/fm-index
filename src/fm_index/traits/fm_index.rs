use std::{char, iter};

use num_traits::Zero;
use pyo3::PyResult;
use rayon::prelude::*;

use super::base_fm_index::BaseFMIndexTrait;
use crate::utils::traits::wavelet_matrix::WaveletMatrixTrait;

pub(crate) trait FMIndexTrait
where
    Self: Sync,
{
    type BaseFMIndex: BaseFMIndexTrait;

    fn get_base_fm_index(&self) -> &Self::BaseFMIndex;

    fn len(&self) -> usize {
        self.get_base_fm_index().len() - 1 // Exclude the sentinel character
    }

    fn max_bit(&self) -> usize {
        self.get_base_fm_index()
            .get_burrows_wheeler_transform()
            .max_bit()
    }

    fn value(&self) -> PyResult<String> {
        let values = self
            .get_base_fm_index()
            .values()?
            .into_iter()
            .filter_map(|c| {
                if c.is_zero() {
                    None
                } else {
                    char::from_u32(c - 1)
                }
            })
            .collect::<String>();

        Ok(values)
    }

    fn range_search(&self, pattern: &str) -> PyResult<(usize, usize)> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        Ok((start, end))
    }

    fn suffix_idx(&self, index: usize) -> PyResult<usize> {
        self.get_base_fm_index().suffix_idx(index)
    }

    fn contains(&self, pattern: &str) -> PyResult<bool> {
        Ok(self.count(pattern)? > 0)
    }

    fn count(&self, pattern: &str) -> PyResult<usize> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        Ok(end - start)
    }

    fn locate(&self, pattern: &str) -> PyResult<Vec<usize>> {
        let (start, end) = self.range_search(pattern)?;
        let result = (start..end)
            .into_par_iter()
            .map(|index| self.get_base_fm_index().suffix_idx(index))
            .collect::<PyResult<_>>()?;

        Ok(result)
    }

    fn starts_with(&self, pattern: &str) -> PyResult<bool> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        Ok(start <= self.get_base_fm_index().get_zero_suffix_idx()
            && self.get_base_fm_index().get_zero_suffix_idx() < end)
    }

    fn ends_with(&self, pattern: &str) -> PyResult<bool> {
        let pattern = pattern
            .chars()
            .map(|c| c as u32 + 1)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        Ok(start != end)
    }
}
