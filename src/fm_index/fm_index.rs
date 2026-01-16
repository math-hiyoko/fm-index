use std::{hash, iter, ops};

use num_traits::{PrimInt, Unsigned};
use pyo3::PyResult;
use rayon::prelude::*;

use super::base_fm_index::BaseFMIndex;
use crate::utils::bit_width::BitWidth;

#[derive(Clone)]
pub(crate) struct FMIndex<
    Element: PrimInt + Unsigned + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync,
> {
    len: usize,
    base_fm_index: BaseFMIndex<Element>,
}

impl<
    Element: PrimInt + Unsigned + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth + Send + Sync,
> FMIndex<Element>
{
    pub(crate) fn new(data: Vec<Element>) -> PyResult<Self> {
        let len = data.len();
        let data = data
            .into_iter()
            .map(|symbol| Some(symbol))
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let base_fm_index = BaseFMIndex::new(data)?;
        Ok(FMIndex { len, base_fm_index })
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
    pub(crate) fn suffix_idx(&self, index: usize) -> PyResult<usize> {
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
    fn test_fm_index_empty() {
        let data = vec![];
        let fm_index = FMIndex::new(data).unwrap();

        assert!(fm_index.len().unwrap().is_zero());
        assert!(fm_index.values().unwrap().is_empty());
        assert!(fm_index.contains(vec![]).unwrap());
        assert!(!fm_index.contains(b"a".to_vec()).unwrap());
        assert_eq!(fm_index.count(vec![]).unwrap(), 1);
        assert!(fm_index.count(b"a".to_vec()).unwrap().is_zero());
        assert_eq!(fm_index.locate(vec![]).unwrap(), [0]);
        assert!(fm_index.locate(b"a".to_vec()).unwrap().is_empty());
        assert!(fm_index.starts_with(vec![]).unwrap());
        assert!(!fm_index.starts_with(b"a".to_vec()).unwrap());
        assert!(fm_index.ends_with(vec![]).unwrap());
        assert!(!fm_index.ends_with(b"a".to_vec()).unwrap());
    }

    #[test]
    fn test_fm_index_single_char() {
        let data = b"aaaaaaaaaa".to_vec();
        let fm_index = FMIndex::new(data.clone()).unwrap();

        assert_eq!(fm_index.len().unwrap(), 10);
        assert_eq!(fm_index.values().unwrap(), data);
        assert!(fm_index.contains(vec![]).unwrap());
        assert!(fm_index.contains(b"a".to_vec()).unwrap());
        assert_eq!(fm_index.count(b"a".to_vec()).unwrap(), 10);
        assert_eq!(
            fm_index.locate(b"a".to_vec()).unwrap(),
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        );
        assert!(fm_index.starts_with(vec![]).unwrap());
        assert!(fm_index.starts_with(b"aa".to_vec()).unwrap());
        assert!(!fm_index.starts_with(b"bb".to_vec()).unwrap());
        assert!(fm_index.ends_with(vec![]).unwrap());
        assert!(fm_index.ends_with(b"aa".to_vec()).unwrap());
        assert!(!fm_index.ends_with(b"bb".to_vec()).unwrap());
    }

    #[test]
    fn test_fm_index_u8() {
        let data = b"mississippi".to_vec();
        let fm_index = FMIndex::new(data.clone()).unwrap();

        assert_eq!(fm_index.len().unwrap(), 11);
        assert_eq!(fm_index.values().unwrap(), data);
        assert!(fm_index.contains(vec![]).unwrap());
        assert!(fm_index.contains(b"is".to_vec()).unwrap());
        assert_eq!(fm_index.count(b"is".to_vec()).unwrap(), 2);
        assert_eq!(fm_index.locate(b"is".to_vec()).unwrap(), [4, 1]);
        assert!(fm_index.starts_with(vec![]).unwrap());
        assert!(fm_index.starts_with(b"mi".to_vec()).unwrap());
        assert!(!fm_index.starts_with(b"si".to_vec()).unwrap());
        assert!(fm_index.ends_with(vec![]).unwrap());
        assert!(fm_index.ends_with(b"pi".to_vec()).unwrap());
        assert!(!fm_index.ends_with(b"ip".to_vec()).unwrap());
    }

    #[test]
    fn test_fm_index_u32() {
        let data = "にわにはにわにわとりがいる"
            .chars()
            .map(|c| c as u32)
            .collect::<Vec<_>>();
        let fm_index = FMIndex::new(data.clone()).unwrap();

        assert_eq!(fm_index.len().unwrap(), 13);
        assert_eq!(fm_index.values().unwrap(), data);
        assert!(fm_index.contains(vec![]).unwrap());
        assert!(
            fm_index
                .contains(['に' as u32, 'わ' as u32].to_vec())
                .unwrap()
        );
        assert_eq!(
            fm_index.count(['に' as u32, 'わ' as u32].to_vec()).unwrap(),
            3
        );
        assert_eq!(
            fm_index
                .locate(['に' as u32, 'わ' as u32].to_vec())
                .unwrap(),
            [6, 0, 4]
        );
        assert!(fm_index.starts_with(vec![]).unwrap());
        assert!(
            fm_index
                .starts_with(['に' as u32, 'わ' as u32].to_vec())
                .unwrap()
        );
        assert!(
            !fm_index
                .starts_with(['い' as u32, 'る' as u32].to_vec())
                .unwrap()
        );
        assert!(fm_index.ends_with(vec![]).unwrap());
        assert!(
            fm_index
                .ends_with(['い' as u32, 'る' as u32].to_vec())
                .unwrap()
        );
        assert!(
            !fm_index
                .ends_with(['に' as u32, 'わ' as u32].to_vec())
                .unwrap()
        );
    }
}
