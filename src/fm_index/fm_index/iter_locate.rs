use std::sync;

use pyo3::{PyResult, prelude::*};

use super::fm_index::FMIndex;

#[pyclass]
pub(crate) struct IterLocate {
    k: usize,
    end: usize,
    fm_index: sync::Arc<FMIndex>,
}

impl IterLocate {
    pub(crate) fn new(pattern: &str, fm_index: sync::Arc<FMIndex>) -> PyResult<Self> {
        let (start, end) = fm_index.range_search(pattern)?;
        Ok(Self {
            k: start,
            end,
            fm_index,
        })
    }
}

#[pymethods]
impl IterLocate {
    pub(crate) fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub(crate) fn __next__(mut slf: PyRefMut<Self>, py: Python<'_>) -> PyResult<Option<usize>> {
        if slf.k >= slf.end {
            return Ok(None);
        }
        let fm_index = slf.fm_index.clone();
        let k = slf.k;
        let result = py.detach(|| fm_index.suffix_idx(k))?;
        slf.k += 1;
        Ok(Some(result))
    }
}
