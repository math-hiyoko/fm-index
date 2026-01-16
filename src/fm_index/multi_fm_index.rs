use std::{collections, hash, iter, ops};

use num_traits::{PrimInt, Unsigned};
use pyo3::PyResult;
use rayon::prelude::*;

use super::base_fm_index::{BaseFMIndex, SUFFIX_ARRAY_SAMPLING_RATE};
use crate::utils::{bit_vector::BitVector, bit_width::BitWidth, suffix_array::suffix_array_option};

#[derive(Clone)]
pub(crate) struct MultiFMIndex<
    Element: PrimInt + Unsigned + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync,
> {
    doc_len: Vec<usize>,
    total_num_chars: usize,
    base_fm_index: BaseFMIndex<Element>,
    doc: collections::HashMap<usize, usize>, // suffix array index -> doc_id
    pos: Vec<(usize, usize)>,                // (doc_id, offset)
}

impl<
    Element: PrimInt + Unsigned + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync,
> MultiFMIndex<Element>
{
    pub(crate) fn new(data: Vec<Vec<Element>>) -> PyResult<Self> {
        let doc_len = data.iter().map(|data| data.len()).collect::<Vec<_>>();
        let total_num_chars = doc_len.iter().sum::<usize>();

        let data = data
            .into_iter()
            .flat_map(|doc| {
                doc.into_iter()
                    .map(|symbol| Some(symbol))
                    .chain(iter::once(None))
            })
            .collect::<Vec<_>>();

        let suffix_idx = suffix_array_option(data.clone());

        let base_fm_index = BaseFMIndex::new_with_suffix_array(data.clone(), suffix_idx.clone())?;

        let data_none_bitvector = BitVector::new(
            data.into_par_iter()
                .map(|value| value.is_none())
                .collect::<Vec<_>>(),
        )?;

        let doc = (1..=doc_len.len())
            .into_par_iter()
            .map(|idx| {
                let k = base_fm_index
                    .burrows_wheeler_transform()
                    .select(None, idx)?
                    .unwrap();
                let doc_id = data_none_bitvector.rank(true, suffix_idx[k])?;
                Ok((k, doc_id))
            })
            .collect::<PyResult<collections::HashMap<usize, usize>>>()?;

        let pos = suffix_idx
            .into_par_iter()
            .step_by(SUFFIX_ARRAY_SAMPLING_RATE)
            .map(|suffix_idx| {
                let doc_id = data_none_bitvector.rank(true, suffix_idx)?;
                let doc_start_idx = if doc_id == 0 {
                    0
                } else {
                    data_none_bitvector.select(true, doc_id)?.unwrap() + 1
                };
                let offset = suffix_idx - doc_start_idx;
                Ok((doc_id, offset))
            })
            .collect::<PyResult<Vec<(usize, usize)>>>()?;

        Ok(MultiFMIndex {
            doc_len,
            total_num_chars,
            base_fm_index,
            doc,
            pos,
        })
    }

    #[inline]
    pub(crate) fn range_search(&self, pattern: Vec<Element>) -> PyResult<(usize, usize)> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok((start, end))
    }

    #[inline]
    pub(crate) fn doc_offset(&self, mut k: usize) -> PyResult<(usize, usize)> {
        let mut step = 0usize;
        loop {
            if let Some(&doc_id) = self.doc.get(&k) {
                let offset = step;
                return Ok((doc_id, offset));
            }
            if k.is_multiple_of(SUFFIX_ARRAY_SAMPLING_RATE) {
                let (doc_id, mut offset) = self.pos[k / SUFFIX_ARRAY_SAMPLING_RATE];
                offset += step;
                return Ok((doc_id, offset));
            }
            step += 1;
            k = self.base_fm_index.lf_mapping(k)?;
        }
    }

    pub(crate) fn len(&self) -> PyResult<usize> {
        Ok(self.doc_len.len())
    }

    pub(crate) fn total_num_chars(&self) -> PyResult<usize> {
        Ok(self.total_num_chars)
    }

    pub(crate) fn max_bit(&self) -> PyResult<usize> {
        self.base_fm_index.burrows_wheeler_transform().max_bit()
    }

    pub(crate) fn values(&self) -> PyResult<Vec<Vec<Element>>> {
        let mut values = self
            .base_fm_index
            .values()?
            .split(|value| value.is_none())
            .map(|slice| slice.iter().filter_map(|&value| value).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        values.truncate(self.len()?); // Remove the last empty slice after the final None

        Ok(values)
    }

    pub(crate) fn contains(&self, pattern: Vec<Element>) -> PyResult<bool> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;
        let bwt = self.base_fm_index.burrows_wheeler_transform();

        Ok(bwt.rank(None, end)? != bwt.rank(None, start)?)
    }

    pub(crate) fn count_all(&self, pattern: Vec<Element>) -> PyResult<usize> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok(end - start)
    }

    pub(crate) fn count(
        &self,
        pattern: Vec<Element>,
    ) -> PyResult<collections::HashMap<usize, usize>> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        let result = (start..end)
            .into_par_iter()
            .map(|k| {
                let (doc_id, _) = self.doc_offset(k)?;
                Ok(doc_id)
            })
            .collect::<PyResult<Vec<usize>>>()?
            .into_iter()
            .fold(
                collections::HashMap::new(),
                |mut acc: collections::HashMap<usize, usize>, doc_id| {
                    *acc.entry(doc_id).or_insert(0) += 1;
                    acc
                },
            );

        Ok(result)
    }

    pub(crate) fn locate(
        &self,
        pattern: Vec<Element>,
    ) -> PyResult<collections::HashMap<usize, Vec<usize>>> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        let result = (start..end)
            .into_par_iter()
            .map(|k| {
                let (doc_id, offset) = self.doc_offset(k)?;
                Ok((doc_id, offset))
            })
            .collect::<PyResult<Vec<(usize, usize)>>>()?
            .into_iter()
            .fold(
                collections::HashMap::new(),
                |mut acc: collections::HashMap<usize, Vec<usize>>, (doc_id, offset)| {
                    acc.entry(doc_id).or_default().push(offset);
                    acc
                },
            );

        Ok(result)
    }

    pub(crate) fn starts_with(&self, pattern: Vec<Element>) -> PyResult<Vec<usize>> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        let mut result = Vec::new();
        if start != end {
            let bwt = self.base_fm_index.burrows_wheeler_transform();
            let start_rank = bwt.rank(None, start)?;
            let end_rank = bwt.rank(None, end)?;
            result = (start_rank + 1..=end_rank)
                .into_par_iter()
                .map(|rank| {
                    let k = bwt.select(None, rank)?.unwrap();
                    let doc_id = self.doc[&k];
                    Ok(doc_id)
                })
                .collect::<PyResult<Vec<usize>>>()?;
        }

        Ok(result)
    }

    pub(crate) fn ends_with(&self, pattern: Vec<Element>) -> PyResult<Vec<usize>> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        let result = (start..end)
            .into_par_iter()
            .map(|k| {
                let (doc_id, _) = self.doc_offset(k)?;
                Ok(doc_id)
            })
            .collect::<PyResult<Vec<usize>>>()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;

    #[test]
    fn test_multi_fm_index_empty() {
        let data = [];
        let fm_index = MultiFMIndex::new(data.to_vec()).unwrap();

        assert!(fm_index.len().unwrap().is_zero());
        assert!(fm_index.values().unwrap().is_empty());
        assert!(!fm_index.contains([].to_vec()).unwrap());
        assert!(!fm_index.contains(b"a".to_vec()).unwrap());
        assert!(fm_index.count_all([].to_vec()).unwrap().is_zero());
        assert!(fm_index.count_all(b"a".to_vec()).unwrap().is_zero());
        assert!(fm_index.count([].to_vec()).unwrap().is_empty());
        assert!(fm_index.count(b"a".to_vec()).unwrap().is_empty());
        assert!(fm_index.locate([].to_vec()).unwrap().is_empty());
        assert!(fm_index.locate(b"a".to_vec()).unwrap().is_empty());
        assert!(fm_index.starts_with([].to_vec()).unwrap().is_empty());
        assert!(fm_index.starts_with(b"a".to_vec()).unwrap().is_empty());
        assert!(fm_index.ends_with([].to_vec()).unwrap().is_empty());
        assert!(fm_index.ends_with(b"a".to_vec()).unwrap().is_empty());
    }

    #[test]
    fn test_multi_fm_index_empties() {
        let data = [vec![], vec![], vec![]];
        let fm_index = MultiFMIndex::new(data.to_vec()).unwrap();

        assert_eq!(fm_index.len().unwrap(), 3);
        assert_eq!(
            fm_index.values().unwrap(),
            [vec![] as Vec<u8>, vec![] as Vec<u8>, vec![] as Vec<u8>]
        );
        assert!(fm_index.contains([].to_vec()).unwrap());
        assert!(!fm_index.contains(b"a".to_vec()).unwrap());
        assert_eq!(fm_index.count_all([].to_vec()).unwrap(), 3);
        assert_eq!(fm_index.count_all(b"a".to_vec()).unwrap(), 0);
        assert_eq!(
            fm_index.count([].to_vec()).unwrap(),
            collections::HashMap::from([(0usize, 1usize), (1usize, 1usize), (2usize, 1usize)])
        );
        assert!(fm_index.count(b"a".to_vec()).unwrap().is_empty());
        assert_eq!(
            fm_index.locate([].to_vec()).unwrap(),
            collections::HashMap::from([(0, vec![0]), (1, vec![0]), (2, vec![0])])
        );
        assert!(fm_index.locate(b"a".to_vec()).unwrap().is_empty());
        assert_eq!(fm_index.starts_with([].to_vec()).unwrap(), [2, 1, 0]);
        assert!(fm_index.starts_with(b"a".to_vec()).unwrap().is_empty());
        assert_eq!(fm_index.ends_with([].to_vec()).unwrap(), [2, 1, 0]);
        assert!(fm_index.ends_with(b"a".to_vec()).unwrap().is_empty());
    }

    #[test]
    fn test_multi_fm_index_single_char() {
        let data = [
            b"aaaaaaaaaa".to_vec(),
            b"".to_vec(),
            b"aaaaaa".to_vec(),
            b"aaaaaaaa".to_vec(),
        ];
        let fm_index = MultiFMIndex::new(data.to_vec()).unwrap();

        assert_eq!(fm_index.len().unwrap(), 4);
        assert_eq!(fm_index.values().unwrap(), data);
        assert!(fm_index.contains([].to_vec()).unwrap());
        assert!(!fm_index.contains(b"a".to_vec()).unwrap());
        assert!(fm_index.contains(b"aaaaaa".to_vec()).unwrap());
        assert_eq!(fm_index.count_all([].to_vec()).unwrap(), 28);
        assert_eq!(fm_index.count_all(b"aa".to_vec()).unwrap(), 21);
        assert_eq!(
            fm_index.count([].to_vec()).unwrap(),
            collections::HashMap::from([(0, 11), (1, 1), (2, 7), (3, 9)])
        );
        assert_eq!(
            fm_index.count(b"aa".to_vec()).unwrap(),
            collections::HashMap::from([(0, 9), (2, 5), (3, 7)])
        );
        assert_eq!(
            fm_index.locate([].to_vec()).unwrap(),
            collections::HashMap::from([
                (0, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
                (1, vec![0]),
                (2, vec![6, 5, 4, 3, 2, 1, 0]),
                (3, vec![8, 7, 6, 5, 4, 3, 2, 1, 0])
            ])
        );
        assert_eq!(
            fm_index.locate(b"aa".to_vec()).unwrap(),
            collections::HashMap::from([
                (0, vec![8, 7, 6, 5, 4, 3, 2, 1, 0]),
                (2, vec![4, 3, 2, 1, 0]),
                (3, vec![6, 5, 4, 3, 2, 1, 0])
            ])
        );
        assert_eq!(fm_index.starts_with([].to_vec()).unwrap(), [1, 2, 3, 0]);
        assert_eq!(fm_index.starts_with(b"aa".to_vec()).unwrap(), [2, 3, 0]);
        assert_eq!(fm_index.ends_with([].to_vec()).unwrap(), [3, 0, 1, 2]);
        assert_eq!(fm_index.ends_with(b"aa".to_vec()).unwrap(), [3, 0, 2]);
    }

    #[test]
    fn test_multi_fm_index_u8() {
        let data = [b"banana".to_vec(), b"bandana".to_vec(), b"anaba".to_vec()];
        let fm_index = MultiFMIndex::new(data.to_vec()).unwrap();

        assert_eq!(fm_index.len().unwrap(), 3);
        assert_eq!(fm_index.values().unwrap(), data);
        assert!(!fm_index.contains([].to_vec()).unwrap());
        assert!(!fm_index.contains(b"ana".to_vec()).unwrap());
        assert!(fm_index.contains(b"banana".to_vec()).unwrap());
        assert_eq!(fm_index.count_all(b"ana".to_vec()).unwrap(), 4);
        assert_eq!(
            fm_index.count(b"ana".to_vec()).unwrap(),
            collections::HashMap::from([(0, 2), (1, 1), (2, 1)])
        );
        assert_eq!(
            fm_index.locate(b"ana".to_vec()).unwrap(),
            collections::HashMap::from([(0, vec![3, 1]), (1, vec![4]), (2, vec![0])])
        );
        assert_eq!(fm_index.starts_with(b"ba".to_vec()).unwrap(), [0, 1]);
        assert_eq!(fm_index.ends_with(b"na".to_vec()).unwrap(), [1, 0]);
    }
}
