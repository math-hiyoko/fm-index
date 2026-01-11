use std::iter;

use pyo3::PyResult;

use super::base_fm_index::BaseFMIndex;

#[derive(Clone)]
pub(crate) struct FMIndex {
    base_fm_index: BaseFMIndex,
}

impl FMIndex {
    pub(crate) fn new(data: &[u8]) -> PyResult<Self> {
        let data = data
            .iter()
            .map(|&symbol| Some(symbol))
            .chain(iter::once(None))
            .collect::<Vec<_>>();
        let base_fm_index = BaseFMIndex::new(&data)?;
        Ok(FMIndex { base_fm_index })
    }

    #[inline]
    pub(crate) fn range_search(&self, pattern: &[u8]) -> PyResult<(usize, usize)> {
        let pattern = pattern
            .iter()
            .map(|&symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(&pattern)?;

        Ok((start, end))
    }

    #[inline]
    pub(crate) fn suffix_idx(&self, index: usize) -> PyResult<usize> {
        self.base_fm_index.suffix_idx(index)
    }

    pub(crate) fn values(&self) -> PyResult<Vec<u8>> {
        let values = self
            .base_fm_index
            .values()?
            .iter()
            .filter_map(|&value| value)
            .collect::<Vec<_>>();

        Ok(values)
    }

    pub(crate) fn contains(&self, pattern: &[u8]) -> PyResult<bool> {
        Ok(self.count(pattern)? > 0)
    }

    pub(crate) fn count(&self, pattern: &[u8]) -> PyResult<usize> {
        let pattern = pattern
            .iter()
            .map(|&symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(&pattern)?;

        Ok(end - start)
    }

    pub(crate) fn locate(&self, pattern: &[u8]) -> PyResult<Vec<usize>> {
        let (start, end) = self.range_search(pattern)?;
        let result = (start..end)
            .map(|index| self.base_fm_index.suffix_idx(index))
            .collect::<PyResult<_>>()?;

        Ok(result)
    }

    pub(crate) fn starts_with(&self, pattern: &[u8]) -> PyResult<bool> {
        let pattern = pattern
            .iter()
            .map(|&symbol| Some(symbol))
            .collect::<Vec<_>>();
        let (start, end) = self.base_fm_index.range_search(&pattern)?;

        Ok(start <= self.base_fm_index.zero_suffix_idx()
            && self.base_fm_index.zero_suffix_idx() < end)
    }

    pub(crate) fn ends_with(&self, pattern: &[u8]) -> PyResult<bool> {
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

        assert!(fm_index.values().unwrap().is_empty());
        assert!(fm_index.contains(&[]).unwrap());
        assert!(!fm_index.contains(b"a").unwrap());
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

        assert_eq!(fm_index.values().unwrap(), data);
        assert!(fm_index.contains(&[]).unwrap());
        assert!(fm_index.contains(b"a").unwrap());
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
    fn test_fm_index() {
        let data = b"mississippi";
        let fm_index = FMIndex::new(data).unwrap();

        assert_eq!(fm_index.values().unwrap(), data);
        assert!(fm_index.contains(&[]).unwrap());
        assert!(fm_index.contains(b"is").unwrap());
        assert_eq!(fm_index.count(b"is").unwrap(), 2);
        assert_eq!(fm_index.locate(b"is").unwrap(), [4, 1]);
        assert!(fm_index.starts_with(&[]).unwrap());
        assert!(fm_index.starts_with(b"mi").unwrap());
        assert!(!fm_index.starts_with(b"si").unwrap());
        assert!(fm_index.ends_with(&[]).unwrap());
        assert!(fm_index.ends_with(b"pi").unwrap());
        assert!(!fm_index.ends_with(b"ip").unwrap());
    }
}
