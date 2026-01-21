use std::{hash, iter, ops};

use num_traits::PrimInt;
use pyo3::PyResult;
use rayon::prelude::*;

use crate::{
    fm_index::base_fm_index::BaseFMIndex,
    utils::{bit_width::BitWidth, suffix_array::suffix_array_option},
};

#[derive(Clone)]
pub(crate) struct FMIndex<
    Element: PrimInt + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync,
> {
    len: usize,
    base_fm_index: BaseFMIndex<Element>,
}

impl<Element: PrimInt + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync>
    FMIndex<Element>
{
    pub(crate) fn new(data: Vec<Element>) -> PyResult<Self> {
        let len = data.len();
        let data = data
            .into_iter()
            .map(|symbol| Some(symbol))
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let suffix_idx = suffix_array_option(&data);
        let base_fm_index = BaseFMIndex::new(data, suffix_idx)?;
        Ok(FMIndex { len, base_fm_index })
    }

    #[inline]
    pub(super) fn range_search(&self, pattern: Vec<Element>) -> PyResult<(usize, usize)> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .collect::<Vec<_>>();
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

    pub(crate) fn values(&self) -> PyResult<Vec<Element>> {
        let values = self
            .base_fm_index
            .values()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        Ok(values)
    }

    pub(crate) fn contains(&self, pattern: Vec<Element>) -> PyResult<bool> {
        Ok(self.count(pattern)? > 0)
    }

    pub(crate) fn count(&self, pattern: Vec<Element>) -> PyResult<usize> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok(end - start)
    }

    pub(crate) fn locate(&self, pattern: Vec<Element>) -> PyResult<Vec<usize>> {
        let (start, end) = self.range_search(pattern)?;
        let result = (start..end)
            .into_par_iter()
            .map(|index| self.base_fm_index.suffix_idx(index))
            .collect::<PyResult<_>>()?;

        Ok(result)
    }

    pub(crate) fn starts_with(&self, pattern: Vec<Element>) -> PyResult<bool> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(pattern)?;

        Ok(start <= self.base_fm_index.zero_suffix_idx()
            && self.base_fm_index.zero_suffix_idx() < end)
    }

    pub(crate) fn ends_with(&self, pattern: Vec<Element>) -> PyResult<bool> {
        let pattern = pattern
            .into_iter()
            .map(|symbol| Some(symbol))
            .chain(iter::once(None))
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
        let data = vec![];
        let index = FMIndex::new(data).unwrap();

        // Length and values
        assert!(index.len().unwrap().is_zero());
        assert!(index.values().unwrap().is_empty());

        // Contains and count
        assert!(index.contains(vec![]).unwrap());
        assert!(!index.contains(b"a".to_vec()).unwrap());
        assert_eq!(index.count(vec![]).unwrap(), 1);
        assert!(index.count(b"a".to_vec()).unwrap().is_zero());

        // Locate
        assert_eq!(index.locate(vec![]).unwrap(), [0]);
        assert!(index.locate(b"a".to_vec()).unwrap().is_empty());

        // Starts with and ends with
        assert!(index.starts_with(vec![]).unwrap());
        assert!(!index.starts_with(b"a".to_vec()).unwrap());
        assert!(index.ends_with(vec![]).unwrap());
        assert!(!index.ends_with(b"a".to_vec()).unwrap());
    }

    #[test]
    fn test_single_repeated_character() {
        let data = b"aaaaaaaaaa".to_vec();
        let index = FMIndex::new(data.clone()).unwrap();

        // Length and values
        assert_eq!(index.len().unwrap(), 10);
        assert_eq!(index.values().unwrap(), data);

        // Contains and count
        assert!(index.contains(vec![]).unwrap());
        assert!(index.contains(b"a".to_vec()).unwrap());
        assert_eq!(index.count(b"a".to_vec()).unwrap(), 10);

        // Locate
        assert_eq!(
            index.locate(b"a".to_vec()).unwrap(),
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        );

        // Starts with and ends with
        assert!(index.starts_with(vec![]).unwrap());
        assert!(index.starts_with(b"aa".to_vec()).unwrap());
        assert!(!index.starts_with(b"bb".to_vec()).unwrap());
        assert!(index.ends_with(vec![]).unwrap());
        assert!(index.ends_with(b"aa".to_vec()).unwrap());
        assert!(!index.ends_with(b"bb".to_vec()).unwrap());
    }

    #[test]
    fn test_byte_string_operations() {
        let data = b"mississippi".to_vec();
        let index = FMIndex::new(data.clone()).unwrap();

        // Length and values
        assert_eq!(index.len().unwrap(), 11);
        assert_eq!(index.values().unwrap(), data);

        // Contains and count
        assert!(index.contains(vec![]).unwrap());
        assert!(index.contains(b"is".to_vec()).unwrap());
        assert_eq!(index.count(b"is".to_vec()).unwrap(), 2);

        // Locate
        assert_eq!(index.locate(b"is".to_vec()).unwrap(), [4, 1]);

        // Starts with
        assert!(index.starts_with(vec![]).unwrap());
        assert!(index.starts_with(b"mi".to_vec()).unwrap());
        assert!(!index.starts_with(b"si".to_vec()).unwrap());

        // Ends with
        assert!(index.ends_with(vec![]).unwrap());
        assert!(index.ends_with(b"pi".to_vec()).unwrap());
        assert!(!index.ends_with(b"ip".to_vec()).unwrap());
    }

    #[test]
    fn test_unicode_string_operations() {
        let text = "にわにはにわにわとりがいる";
        let data = text.chars().map(|c| c as u32).collect::<Vec<_>>();
        let index = FMIndex::new(data.clone()).unwrap();

        let pattern_niwa = vec!['に' as u32, 'わ' as u32];
        let pattern_iru = vec!['い' as u32, 'る' as u32];

        // Length and values
        assert_eq!(index.len().unwrap(), 13);
        assert_eq!(index.values().unwrap(), data);

        // Contains and count
        assert!(index.contains(vec![]).unwrap());
        assert!(index.contains(pattern_niwa.clone()).unwrap());
        assert_eq!(index.count(pattern_niwa.clone()).unwrap(), 3);

        // Locate
        assert_eq!(index.locate(pattern_niwa.clone()).unwrap(), [6, 0, 4]);

        // Starts with
        assert!(index.starts_with(vec![]).unwrap());
        assert!(index.starts_with(pattern_niwa.clone()).unwrap());
        assert!(!index.starts_with(pattern_iru.clone()).unwrap());

        // Ends with
        assert!(index.ends_with(vec![]).unwrap());
        assert!(index.ends_with(pattern_iru).unwrap());
        assert!(!index.ends_with(pattern_niwa).unwrap());
    }
}
