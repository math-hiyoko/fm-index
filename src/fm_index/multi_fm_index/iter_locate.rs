use std::sync;

use pyo3::{PyResult, exceptions::PyValueError, prelude::*};

use super::multi_fm_index::MultiFMIndex;

#[pyclass]
pub(crate) struct IterLocate {
    doc_id: Option<usize>,
    k: usize,
    end: usize,
    multi_fm_index: sync::Arc<MultiFMIndex>,
}

impl IterLocate {
    pub(crate) fn new(
        doc_id: Option<usize>,
        pattern: &str,
        multi_fm_index: sync::Arc<MultiFMIndex>,
    ) -> PyResult<Self> {
        let (start, end) = multi_fm_index.range_search(pattern)?;
        match doc_id {
            Some(doc_id) => {
                if doc_id >= multi_fm_index.num_docs()? {
                    return Err(PyValueError::new_err("doc_id is out of bounds"));
                }
                let rank = multi_fm_index
                    .doc_id_of_index()
                    .rank(doc_id as u32, start)?;
                let start = multi_fm_index
                    .doc_id_of_index()
                    .select(doc_id as u32, rank + 1)?
                    .unwrap_or(end);
                Ok(Self {
                    doc_id: Some(doc_id),
                    k: start,
                    end,
                    multi_fm_index,
                })
            }
            None => Ok(Self {
                doc_id,
                k: start,
                end,
                multi_fm_index,
            }),
        }
    }
}

#[pymethods]
impl IterLocate {
    pub(crate) fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub(crate) fn __next__(mut slf: PyRefMut<Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if slf.k >= slf.end {
            return Ok(None);
        }
        let multi_fm_index = slf.multi_fm_index.clone();
        let k = slf.k;
        let result = match slf.doc_id {
            Some(_) => {
                let (_, offset) = py.detach(|| multi_fm_index.doc_offset(k))?;
                offset.into_pyobject(py)?.unbind().into()
            }
            None => {
                let (doc_id, offset) = py.detach(|| multi_fm_index.doc_offset(k))?;
                (doc_id, offset).into_pyobject(py)?.unbind().into()
            }
        };
        slf.k = match slf.doc_id {
            Some(doc_id) => {
                let next_k = py.detach(|| {
                    let rank = multi_fm_index
                        .doc_id_of_index()
                        .rank(doc_id as u32, k + 1)?;
                    multi_fm_index
                        .doc_id_of_index()
                        .select(doc_id as u32, rank + 1)
                })?;
                next_k.unwrap_or(slf.end)
            }
            None => k + 1,
        };
        Ok(Some(result))
    }
}
