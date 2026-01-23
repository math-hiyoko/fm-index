use std::{char, iter};

use num_traits::Zero;
use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{fm_index::base_fm_index::BaseFMIndex, utils::suffix_array::suffix_array};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct FMIndex {
    len: usize,
    base_fm_index: BaseFMIndex,
}

impl FMIndex {
    pub(crate) fn new(data: &str) -> PyResult<Self> {
        let len = data.chars().count();
        let data = data
            .chars()
            .map(|c| c as u32 + 1)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let suffix_idx = suffix_array(&data);
        let base_fm_index = BaseFMIndex::new(data, suffix_idx)?;
        Ok(FMIndex { len, base_fm_index })
    }

    #[inline]
    pub(super) fn range_search(&self, pattern: &str) -> PyResult<(usize, usize)> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok((start, end))
    }

    #[inline]
    pub(super) fn suffix_idx(&self, index: usize) -> PyResult<usize> {
        self.base_fm_index.suffix_idx(index)
    }

    pub(crate) fn len(&self) -> PyResult<usize> {
        Ok(self.len)
    }

    pub(crate) fn max_bit(&self) -> PyResult<usize> {
        self.base_fm_index.burrows_wheeler_transform().max_bit()
    }

    pub(crate) fn value(&self) -> PyResult<String> {
        let values = self
            .base_fm_index
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

    pub(crate) fn contains(&self, pattern: &str) -> PyResult<bool> {
        Ok(self.count(pattern)? > 0)
    }

    pub(crate) fn count(&self, pattern: &str) -> PyResult<usize> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok(end - start)
    }

    pub(crate) fn locate(&self, pattern: &str) -> PyResult<Vec<usize>> {
        let (start, end) = self.range_search(pattern)?;
        let result = (start..end)
            .into_par_iter()
            .map(|index| self.base_fm_index.suffix_idx(index))
            .collect::<PyResult<_>>()?;

        Ok(result)
    }

    pub(crate) fn starts_with(&self, pattern: &str) -> PyResult<bool> {
        let pattern = pattern.chars().map(|c| c as u32 + 1).collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok(start <= self.base_fm_index.zero_suffix_idx()
            && self.base_fm_index.zero_suffix_idx() < end)
    }

    pub(crate) fn ends_with(&self, pattern: &str) -> PyResult<bool> {
        let pattern = pattern
            .chars()
            .map(|c| c as u32 + 1)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok(start != end)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;

    #[test]
    fn test_empty_index() {
        let data = "";
        let index = FMIndex::new(data).unwrap();

        // Length and values
        assert!(index.len().unwrap().is_zero());
        assert!(index.value().unwrap().is_empty());

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(!index.contains("a").unwrap());
        assert_eq!(index.count("").unwrap(), 1);
        assert!(index.count("a").unwrap().is_zero());

        // Locate
        assert_eq!(index.locate("").unwrap(), [0]);
        assert!(index.locate("a").unwrap().is_empty());

        // Starts with and ends with
        assert!(index.starts_with("").unwrap());
        assert!(!index.starts_with("a").unwrap());
        assert!(index.ends_with("").unwrap());
        assert!(!index.ends_with("a").unwrap());
    }

    #[test]
    fn test_single_repeated_character() {
        let data = "aaaaaaaaaa";
        let index = FMIndex::new(data).unwrap();

        // Length and values
        assert_eq!(index.len().unwrap(), 10);
        assert_eq!(index.value().unwrap(), data);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(index.contains("a").unwrap());
        assert_eq!(index.count("a").unwrap(), 10);

        // Locate
        assert_eq!(
            {
                let mut sorted = index.locate("a").unwrap();
                sorted.sort();
                sorted
            },
            (0..10).collect::<Vec<_>>()
        );

        // Starts with and ends with
        assert!(index.starts_with("").unwrap());
        assert!(index.starts_with("aa").unwrap());
        assert!(!index.starts_with("bb").unwrap());
        assert!(index.ends_with("").unwrap());
        assert!(index.ends_with("aa").unwrap());
        assert!(!index.ends_with("bb").unwrap());
    }

    #[test]
    fn test_byte_string_operations() {
        let data = "mississippi";
        let index = FMIndex::new(data).unwrap();

        // Length and values
        assert_eq!(index.len().unwrap(), 11);
        assert_eq!(index.value().unwrap(), data);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(index.contains("is").unwrap());
        assert_eq!(index.count("is").unwrap(), 2);

        // Locate
        assert_eq!(index.locate("is").unwrap(), [4, 1]);

        // Starts with
        assert!(index.starts_with("").unwrap());
        assert!(index.starts_with("mi").unwrap());
        assert!(!index.starts_with("si").unwrap());

        // Ends with
        assert!(index.ends_with("").unwrap());
        assert!(index.ends_with("pi").unwrap());
        assert!(!index.ends_with("ip").unwrap());
    }

    #[test]
    fn test_unicode_string_operations() {
        let text = "にわにはにわにわとりがいる";
        let index = FMIndex::new(text).unwrap();

        // Length and values
        assert_eq!(index.len().unwrap(), 13);
        assert_eq!(index.value().unwrap(), text);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(index.contains("にわ").unwrap());
        assert_eq!(index.count("にわ").unwrap(), 3);

        // Locate
        assert_eq!(index.locate("にわ").unwrap(), [6, 0, 4]);

        // Starts with
        assert!(index.starts_with("").unwrap());
        assert!(index.starts_with("にわ").unwrap());
        assert!(!index.starts_with("いる").unwrap());

        // Ends with
        assert!(index.ends_with("").unwrap());
        assert!(index.ends_with("いる").unwrap());
        assert!(!index.ends_with("にわ").unwrap());
    }
}
