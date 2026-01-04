use std::{hash, iter, ops};

use num_traits::{PrimInt, Unsigned};
use pyo3::PyResult;

use super::base_fm_index::BaseFMIndex;
use crate::utils::bit_width::BitWidth;

#[derive(Clone)]
pub(crate) struct FMIndex<
    Element: PrimInt + Unsigned + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth,
> {
    len: usize,
    base_fm_index: BaseFMIndex<Element>,
}

impl<Element: PrimInt + Unsigned + hash::Hash + ops::BitOrAssign + ops::ShlAssign + BitWidth>
    FMIndex<Element>
{
    pub(crate) fn new(data: &[Element]) -> PyResult<Self> {
        let len = data.len();
        let data = data
            .iter()
            .map(|&symbol| Some(symbol))
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let base_fm_index = BaseFMIndex::new(&data)?;
        Ok(FMIndex { len, base_fm_index })
    }

    pub(crate) fn len(&self) -> PyResult<usize> {
        Ok(self.len)
    }

    pub(crate) fn values(&self) -> PyResult<Vec<Element>> {
        let values = self
            .base_fm_index
            .values()?
            .iter()
            .filter_map(|&value| value)
            .collect::<Vec<_>>();

        Ok(values)
    }

    pub(crate) fn count(&self, pattern: &[Element]) -> PyResult<usize> {
        let pattern = pattern
            .iter()
            .map(|&symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(&pattern)?;

        Ok(end - start)
    }

    pub(crate) fn locate(&self, pattern: &[Element]) -> PyResult<Vec<usize>> {
        let pattern = pattern
            .iter()
            .map(|&symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(&pattern)?;
        let result = (start..end)
            .map(|index| self.base_fm_index.suffix_idx(index))
            .collect::<PyResult<_>>()?;

        Ok(result)
    }

    pub(crate) fn starts_with(&self, pattern: &[Element]) -> PyResult<bool> {
        let pattern = pattern
            .iter()
            .map(|&symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(&pattern)?;

        Ok(start <= self.base_fm_index.zero_suffix_idx()
            && self.base_fm_index.zero_suffix_idx() < end)
    }

    pub(crate) fn ends_with(&self, pattern: &[Element]) -> PyResult<bool> {
        let pattern = pattern
            .iter()
            .map(|&symbol| Some(symbol))
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(&pattern)?;

        Ok(start != end)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;

    #[test]
    fn test_fm_index_empty() {
        let data = [];
        let fm_index = FMIndex::new(&data).unwrap();

        assert!(fm_index.len().unwrap().is_zero());
        assert!(fm_index.values().unwrap().is_empty());
        assert_eq!(fm_index.count(&[]).unwrap(), 1);
        assert!(fm_index.count(b"a").unwrap().is_zero());
        assert_eq!(fm_index.locate(&[]).unwrap(), [0]);
        assert!(fm_index.locate(b"a").unwrap().is_empty());
        assert!(fm_index.starts_with(&[]).unwrap());
        assert!(!fm_index.starts_with(b"a").unwrap());
        assert!(fm_index.ends_with(&[]).unwrap());
        assert!(!fm_index.ends_with(b"a").unwrap());
    }

    #[test]
    fn test_fm_index_single_char() {
        let data = b"aaaaaaaaaa";
        let fm_index = FMIndex::new(data).unwrap();

        assert_eq!(fm_index.len().unwrap(), 10);
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(fm_index.count(b"a").unwrap(), 10);
        assert_eq!(
            fm_index.locate(b"a").unwrap(),
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        );
        assert!(fm_index.starts_with(&[]).unwrap());
        assert!(fm_index.starts_with(b"aa").unwrap());
        assert!(!fm_index.starts_with(b"bb").unwrap());
        assert!(fm_index.ends_with(&[]).unwrap());
        assert!(fm_index.ends_with(b"aa").unwrap());
        assert!(!fm_index.ends_with(b"bb").unwrap());
    }

    #[test]
    fn test_fm_index_u8() {
        let data = b"mississippi";
        let fm_index = FMIndex::new(data).unwrap();

        assert_eq!(fm_index.len().unwrap(), 11);
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(fm_index.count(b"is").unwrap(), 2);
        assert_eq!(fm_index.locate(b"is").unwrap(), [4, 1]);
        assert!(fm_index.starts_with(&[]).unwrap());
        assert!(fm_index.starts_with(b"mi").unwrap());
        assert!(!fm_index.starts_with(b"si").unwrap());
        assert!(fm_index.ends_with(&[]).unwrap());
        assert!(fm_index.ends_with(b"pi").unwrap());
        assert!(!fm_index.ends_with(b"ip").unwrap());
    }

    #[test]
    fn test_fm_index_u32() {
        let data = "にわにはにわにわとりがいる"
            .chars()
            .map(|c| c as u32)
            .collect::<Vec<_>>();
        let fm_index = FMIndex::new(&data).unwrap();

        assert_eq!(fm_index.len().unwrap(), 13);
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(fm_index.count(&['に' as u32, 'わ' as u32]).unwrap(), 3);
        assert_eq!(
            fm_index.locate(&['に' as u32, 'わ' as u32]).unwrap(),
            [6, 0, 4]
        );
        assert!(fm_index.starts_with(&[]).unwrap());
        assert!(fm_index.starts_with(&['に' as u32, 'わ' as u32]).unwrap());
        assert!(!fm_index.starts_with(&['い' as u32, 'る' as u32]).unwrap());
        assert!(fm_index.ends_with(&[]).unwrap());
        assert!(fm_index.ends_with(&['い' as u32, 'る' as u32]).unwrap());
        assert!(!fm_index.ends_with(&['に' as u32, 'わ' as u32]).unwrap());
    }
}
