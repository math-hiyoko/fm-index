use std::{collections, iter};

use num_traits::Zero;
use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::fm_index::base_fm_index::BaseFMIndex;
use crate::utils::{suffix_array::suffix_array, wavelet_matrix::WaveletMatrix};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct MultiFMIndex {
    doc_len: Vec<usize>,
    total_num_chars: usize,
    doc_start_index: Vec<usize>,
    doc_id_of_index: WaveletMatrix,
    base_fm_index: BaseFMIndex,
}

impl MultiFMIndex {
    pub(crate) fn new(data: Vec<String>) -> PyResult<Self> {
        let doc_len = data
            .iter()
            .map(|data| data.chars().count())
            .collect::<Vec<_>>();
        let total_num_chars = doc_len.iter().sum::<usize>();

        let data = data
            .into_iter()
            .flat_map(|doc| {
                doc.chars()
                    .map(|c| c as u32 + 1)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .chain(iter::once(0))
            })
            .collect::<Vec<_>>();

        let suffix_idx = suffix_array(&data);

        let doc_start_index = doc_len
            .iter()
            .scan(0usize, |acc, &len| {
                let start = *acc;
                *acc += len + 1; // +1 for the delimiter
                Some(start)
            })
            .collect::<Vec<_>>();

        let doc_id_of_index = {
            let doc_ids = data
                .iter()
                .scan(0u32, |doc_id, &value| {
                    let ret = *doc_id;
                    if value.is_zero() {
                        *doc_id += 1;
                    }
                    Some(ret)
                })
                .collect::<Vec<_>>();
            println!("doc_ids: {:?}", doc_ids);

            WaveletMatrix::new(
                suffix_idx
                    .iter()
                    .map(|&idx| doc_ids[idx])
                    .collect::<Vec<_>>(),
            )?
        };

        let base_fm_index = BaseFMIndex::new(data, suffix_idx)?;

        Ok(MultiFMIndex {
            doc_len,
            total_num_chars,
            doc_start_index,
            doc_id_of_index,
            base_fm_index,
        })
    }

    #[inline]
    pub(super) fn range_search(&self, pattern: &str) -> PyResult<(usize, usize)> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok((start, end))
    }

    #[inline]
    pub(super) fn doc_offset(&self, k: usize) -> PyResult<(usize, usize)> {
        let doc_id = self.doc_id_of_index.access(k)? as usize;
        let doc_start = self.doc_start_index[doc_id];
        let offset = self.base_fm_index.suffix_idx(k)? - doc_start;
        Ok((doc_id, offset))
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

    pub(crate) fn values(&self) -> PyResult<Vec<String>> {
        let mut values = self
            .base_fm_index
            .values()?
            .split(|value| value.is_zero())
            .map(|slice| {
                slice
                    .iter()
                    .map(|&c| char::from_u32(c - 1).unwrap())
                    .collect()
            })
            .collect::<Vec<_>>();
        values.truncate(self.len()?); // Remove the last empty slice after the final 0

        Ok(values)
    }

    pub(crate) fn contains(&self, pattern: &str) -> PyResult<bool> {
        let pattern = pattern
            .chars()
            .map(|c| c as u32 + 1)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;
        let bwt = self.base_fm_index.burrows_wheeler_transform();

        Ok(bwt.rank(0u32, end)? != bwt.rank(0u32, start)?)
    }

    pub(crate) fn count_all(&self, pattern: &str) -> PyResult<usize> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok(end - start)
    }

    pub(crate) fn count(&self, pattern: &str) -> PyResult<collections::HashMap<usize, usize>> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        let result = self
            .doc_id_of_index
            .range_list(start, end)?
            .into_iter()
            .map(|(doc_id, count)| (doc_id as usize, count))
            .collect::<collections::HashMap<usize, usize>>();

        Ok(result)
    }

    pub(crate) fn locate(
        &self,
        pattern: &str,
    ) -> PyResult<collections::HashMap<usize, Vec<usize>>> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
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
                collections::HashMap::<usize, Vec<usize>>::new(),
                |mut acc, (doc_id, offset)| {
                    acc.entry(doc_id).or_default().push(offset);
                    acc
                },
            );

        Ok(result)
    }

    pub(crate) fn starts_with(&self, pattern: &str) -> PyResult<Vec<usize>> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        let mut result = vec![];
        if start != end {
            let bwt = self.base_fm_index.burrows_wheeler_transform();
            let start_rank = bwt.rank(0, start)?;
            let end_rank = bwt.rank(0, end)?;
            result = (start_rank..end_rank)
                .into_par_iter()
                .map(|rank| {
                    let k = bwt.select(0, rank + 1)?.unwrap();
                    let doc_id = self.doc_id_of_index.access(k)? as usize;
                    Ok(doc_id)
                })
                .collect::<PyResult<Vec<_>>>()?;
        }

        Ok(result)
    }

    pub(crate) fn ends_with(&self, pattern: &str) -> PyResult<Vec<usize>> {
        let pattern = pattern
            .chars()
            .map(|c| c as u32 + 1)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

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

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;

    #[test]
    fn test_empty_collection() {
        let data = Vec::<String>::new();
        let index = MultiFMIndex::new(data).unwrap();

        // Length and values
        assert!(index.len().unwrap().is_zero());
        assert!(index.values().unwrap().is_empty());

        // Contains and count
        assert!(!index.contains("").unwrap());
        assert!(!index.contains("a").unwrap());
        assert!(index.count_all("").unwrap().is_zero());
        assert!(index.count_all("a").unwrap().is_zero());
        assert!(index.count("").unwrap().is_empty());
        assert!(index.count("a").unwrap().is_empty());

        // Locate
        assert!(index.locate("").unwrap().is_empty());
        assert!(index.locate("a").unwrap().is_empty());

        // Starts with and ends with
        assert!(index.starts_with("").unwrap().is_empty());
        assert!(index.starts_with("a").unwrap().is_empty());
        assert!(index.ends_with("").unwrap().is_empty());
        assert!(index.ends_with("a").unwrap().is_empty());
    }

    #[test]
    fn test_collection_of_empty_documents() {
        let data = vec!["".to_string(), "".to_string(), "".to_string()];
        let index = MultiFMIndex::new(data).unwrap();

        let expected_values: Vec<String> = vec!["".to_string(), "".to_string(), "".to_string()];

        // Length and values
        assert_eq!(index.len().unwrap(), 3);
        assert_eq!(index.values().unwrap(), expected_values);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(!index.contains("a").unwrap());
        assert_eq!(index.count_all("").unwrap(), 3);
        assert_eq!(index.count_all("a").unwrap(), 0);
        assert_eq!(
            index.count("").unwrap(),
            collections::HashMap::from([(0, 1), (1, 1), (2, 1)])
        );
        assert!(index.count("a").unwrap().is_empty());

        // Locate
        assert_eq!(
            index.locate("").unwrap(),
            collections::HashMap::from([(0, vec![0]), (1, vec![0]), (2, vec![0])])
        );
        assert!(index.locate("a").unwrap().is_empty());

        // Starts with and ends with
        assert_eq!(index.starts_with("").unwrap(), [2, 1, 0]);
        assert!(index.starts_with("a").unwrap().is_empty());
        assert_eq!(index.ends_with("").unwrap(), [2, 1, 0]);
        assert!(index.ends_with("a").unwrap().is_empty());
    }

    #[test]
    fn test_single_repeated_character_documents() {
        let data = vec![
            "aaaaaaaaaa".to_string(),
            "".to_string(),
            "aaaaaa".to_string(),
            "aaaaaaaa".to_string(),
        ];
        let index = MultiFMIndex::new(data).unwrap();

        let expected_values = vec![
            "aaaaaaaaaa".to_string(),
            "".to_string(),
            "aaaaaa".to_string(),
            "aaaaaaaa".to_string(),
        ];

        // Length and values
        assert_eq!(index.len().unwrap(), 4);
        assert_eq!(index.values().unwrap(), expected_values);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(!index.contains("a").unwrap());
        assert!(index.contains("aaaaaa").unwrap());
        assert_eq!(index.count_all("").unwrap(), 28);
        assert_eq!(index.count_all("aa").unwrap(), 21);
        assert_eq!(
            index.count("").unwrap(),
            collections::HashMap::from([(0, 11), (1, 1), (2, 7), (3, 9)])
        );
        assert_eq!(
            index.count("aa").unwrap(),
            collections::HashMap::from([(0, 9), (2, 5), (3, 7)])
        );

        // Locate
        assert_eq!(
            index.locate("").unwrap(),
            collections::HashMap::from([
                (0, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
                (1, vec![0]),
                (2, vec![6, 5, 4, 3, 2, 1, 0]),
                (3, vec![8, 7, 6, 5, 4, 3, 2, 1, 0])
            ])
        );
        assert_eq!(
            index.locate("aa").unwrap(),
            collections::HashMap::from([
                (0, vec![8, 7, 6, 5, 4, 3, 2, 1, 0]),
                (2, vec![4, 3, 2, 1, 0]),
                (3, vec![6, 5, 4, 3, 2, 1, 0])
            ])
        );

        // Starts with and ends with
        assert_eq!(index.starts_with("").unwrap(), [1, 2, 3, 0]);
        assert_eq!(index.starts_with("aa").unwrap(), [2, 3, 0]);
        assert_eq!(index.ends_with("").unwrap(), [3, 0, 1, 2]);
        assert_eq!(index.ends_with("aa").unwrap(), [3, 0, 2]);
    }

    #[test]
    fn test_multiple_byte_string_documents() {
        let data = vec![
            "banana".to_string(),
            "bandana".to_string(),
            "anaba".to_string(),
        ];
        let index = MultiFMIndex::new(data).unwrap();

        let expected_values = vec![
            "banana".to_string(),
            "bandana".to_string(),
            "anaba".to_string(),
        ];

        // Length and values
        assert_eq!(index.len().unwrap(), 3);
        assert_eq!(index.values().unwrap(), expected_values);

        // Contains and count
        assert!(!index.contains("").unwrap());
        assert!(!index.contains("ana").unwrap());
        assert!(index.contains("banana").unwrap());
        assert_eq!(index.count_all("ana").unwrap(), 4);
        assert_eq!(
            index.count("ana").unwrap(),
            collections::HashMap::from([(0, 2), (1, 1), (2, 1)])
        );

        // Locate
        assert_eq!(
            index.locate("ana").unwrap(),
            collections::HashMap::from([(0, vec![3, 1]), (1, vec![4]), (2, vec![0])])
        );

        // Starts with and ends with
        assert_eq!(index.starts_with("ba").unwrap(), [0, 1]);
        assert_eq!(index.ends_with("na").unwrap(), [1, 0]);
    }
}
