use std::sync;

use pyo3::{PyResult, prelude::*};

use super::multi_fm_index::MultiFMIndex;

#[pyclass]
pub(crate) struct IterLocate {
    k: usize,
    end: usize,
    multi_fm_index: sync::Arc<MultiFMIndex>,
}

impl IterLocate {
    pub(crate) fn new(pattern: &str, multi_fm_index: sync::Arc<MultiFMIndex>) -> PyResult<Self> {
        let (start, end) = multi_fm_index.range_search(pattern)?;
        Ok(Self {
            k: start,
            end,
            multi_fm_index: multi_fm_index.clone(),
        })
    }
}

#[pymethods]
impl IterLocate {
    pub(crate) fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub(crate) fn __next__(
        mut slf: PyRefMut<Self>,
        py: Python<'_>,
    ) -> PyResult<Option<(usize, usize)>> {
        if slf.k >= slf.end {
            return Ok(None);
        }
        let multi_fm_index = slf.multi_fm_index.clone();
        let k = slf.k;
        let result = py.detach(|| multi_fm_index.doc_offset(k))?;
        slf.k += 1;
        Ok(Some(result))
    }
}
