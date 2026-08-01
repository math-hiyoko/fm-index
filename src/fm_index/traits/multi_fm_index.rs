use std::{collections, iter};

use num_traits::Zero;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

use super::base_fm_index::BaseFMIndexTrait;

use crate::utils::traits::{bit_vector::BitVectorTrait, wavelet_matrix::WaveletMatrixTrait};

pub(crate) trait MultiFMIndexTrait
where
    Self: Sync,
{
    type BitVector: BitVectorTrait;
    type WaveletMatrix: WaveletMatrixTrait<usize, Self::BitVector>;
    type BaseFMIndex: BaseFMIndexTrait;

    fn get_num_docs(&self) -> usize;

    fn get_base_fm_index(&self) -> &Self::BaseFMIndex;

    fn get_doc_start_index(&self) -> &[usize];

    fn get_doc_id_of_index(&self) -> &Self::WaveletMatrix;

    fn total_num_chars(&self) -> usize {
        self.get_base_fm_index().len() - self.get_num_docs()
    }

    fn max_bit(&self) -> usize {
        self.get_base_fm_index()
            .get_burrows_wheeler_transform()
            .max_bit()
    }

    fn values(&self) -> PyResult<Vec<String>> {
        let mut values = self
            .get_base_fm_index()
            .values()?
            .split(|value| value.is_zero())
            .map(|slice| {
                slice
                    .iter()
                    .map(|&c| char::from_u32(c - 1).unwrap())
                    .collect()
            })
            .collect::<Vec<_>>();
        values.truncate(self.get_num_docs()); // Remove the last empty slice after the final 0

        Ok(values)
    }

    fn range_search(&self, pattern: &str) -> PyResult<(usize, usize)> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        Ok((start, end))
    }

    fn doc_offset(&self, k: usize) -> PyResult<(usize, usize)> {
        let doc_id = self.get_doc_id_of_index().access(k)? as usize;
        let doc_start = self.get_doc_start_index()[doc_id];
        let offset = self.get_base_fm_index().suffix_idx(k)? - doc_start;
        Ok((doc_id, offset))
    }

    fn contains(&self, pattern: &str) -> PyResult<bool> {
        let pattern = pattern
            .chars()
            .map(|c| c as u32 + 1)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        Ok(self
            .get_base_fm_index()
            .get_burrows_wheeler_transform()
            .rank(0u32, end)?
            != self
                .get_base_fm_index()
                .get_burrows_wheeler_transform()
                .rank(0u32, start)?)
    }

    fn count_all(&self, pattern: &str) -> PyResult<usize> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        Ok(end - start)
    }

    fn count(&self, pattern: &str) -> PyResult<collections::HashMap<usize, usize>> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        let result = self
            .get_doc_id_of_index()
            .range_list(start, end)?
            .into_iter()
            .map(|(doc_id, count)| (doc_id as usize, count))
            .collect::<collections::HashMap<usize, usize>>();

        Ok(result)
    }

    fn count_within_doc(&self, doc_id: usize, pattern: &str) -> PyResult<usize> {
        if doc_id >= self.get_num_docs() {
            return Err(PyValueError::new_err("doc_id is out of bounds"));
        }
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        let count_within_doc = self.get_doc_id_of_index().rank(doc_id, end)?
            - self.get_doc_id_of_index().rank(doc_id, start)?;

        Ok(count_within_doc)
    }

    fn topk(&self, pattern: &str, k: usize) -> PyResult<Vec<(usize, usize)>> {
        if k.is_zero() {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        // If no matches found, return empty result
        if start >= end {
            return Ok(Vec::new());
        }

        let result = self
            .get_doc_id_of_index()
            .topk(start, end, k)?
            .into_iter()
            .map(|(doc_id, count)| (doc_id as usize, count))
            .collect::<Vec<_>>();

        Ok(result)
    }

    fn locate(&self, pattern: &str) -> PyResult<collections::HashMap<usize, Vec<usize>>> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        let result = (start..end)
            .into_par_iter()
            .map(|k| {
                let (doc_id, offset) = self.doc_offset(k)?;
                Ok((doc_id, offset))
            })
            .collect::<PyResult<Vec<(usize, usize)>>>()?
            .into_iter()
            .fold(
                collections::HashMap::<usize, Vec<usize>>::new(),
                |mut acc, (doc_id, offset)| {
                    acc.entry(doc_id).or_default().push(offset);
                    acc
                },
            );

        Ok(result)
    }

    fn locate_within_doc(&self, doc_id: usize, pattern: &str) -> PyResult<Vec<usize>> {
        if doc_id >= self.get_num_docs() {
            return Err(PyValueError::new_err("doc_id is out of bounds"));
        }
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        let start_index = self.get_doc_start_index()[doc_id];
        let start_rank = self.get_doc_id_of_index().rank(doc_id, start)?;
        let end_rank = self.get_doc_id_of_index().rank(doc_id, end)?;
        let result = (start_rank..end_rank)
            .into_par_iter()
            .map(|rank| {
                let k = self
                    .get_doc_id_of_index()
                    .select(doc_id, rank + 1)?
                    .unwrap();
                let offset = self.get_base_fm_index().suffix_idx(k)? - start_index;
                Ok(offset)
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(result)
    }

    fn starts_with(&self, pattern: &str) -> PyResult<Vec<usize>> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        let mut result = vec![];
        if start != end {
            let start_rank = self
                .get_base_fm_index()
                .get_burrows_wheeler_transform()
                .rank(0, start)?;
            let end_rank = self
                .get_base_fm_index()
                .get_burrows_wheeler_transform()
                .rank(0, end)?;
            result = (start_rank..end_rank)
                .into_par_iter()
                .map(|rank| {
                    let k = self
                        .get_base_fm_index()
                        .get_burrows_wheeler_transform()
                        .select(0, rank + 1)?
                        .unwrap();
                    let doc_id = self.get_doc_id_of_index().access(k)? as usize;
                    Ok(doc_id)
                })
                .collect::<PyResult<Vec<_>>>()?;
        }

        Ok(result)
    }

    fn ends_with(&self, pattern: &str) -> PyResult<Vec<usize>> {
        let pattern = pattern
            .chars()
            .map(|c| c as u32 + 1)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let (start, end) = self.get_base_fm_index().range_search(pattern)?;

        let result = (start..end)
            .into_par_iter()
            .map(|k| {
                let (doc_id, _) = self.doc_offset(k)?;
                Ok(doc_id)
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(result)
    }
}
