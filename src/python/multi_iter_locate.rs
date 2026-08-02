use pyo3::{PyResult, exceptions::PyValueError, prelude::*};

use super::multi_fm_index::MultiFMIndexEnum;
use crate::{
    fm_index::traits::multi_fm_index::MultiFMIndexTrait,
    utils::traits::wavelet_matrix::WaveletMatrixTrait,
};

#[pyclass]
pub(super) struct IterLocate {
    doc_id: Option<usize>,
    k: usize,
    end: usize,
    multi_fm_index: MultiFMIndexEnum,
}

impl IterLocate {
    pub(super) fn new(
        doc_id: Option<usize>,
        pattern: &str,
        multi_fm_index: MultiFMIndexEnum,
    ) -> PyResult<Self> {
        let (mut start, end) = match &multi_fm_index {
            MultiFMIndexEnum::InMemory(multi_fm_index) => multi_fm_index.range_search(pattern)?,
            MultiFMIndexEnum::OnDisk(disk_multi_fm_index) => {
                disk_multi_fm_index.range_search(pattern)?
            }
        };
        if let Some(doc_id) = doc_id {
            match &multi_fm_index {
                MultiFMIndexEnum::InMemory(multi_fm_index) => {
                    if doc_id >= multi_fm_index.get_num_docs() {
                        return Err(PyValueError::new_err("doc_id is out of bounds"));
                    }
                    let rank = multi_fm_index.get_doc_id_of_index().rank(doc_id, start)?;
                    start = multi_fm_index
                        .get_doc_id_of_index()
                        .select(doc_id, rank + 1)?
                        .unwrap_or(end);
                }
                MultiFMIndexEnum::OnDisk(disk_multi_fm_index) => {
                    if doc_id >= disk_multi_fm_index.get_num_docs() {
                        return Err(PyValueError::new_err("doc_id is out of bounds"));
                    }
                    let rank = disk_multi_fm_index
                        .get_doc_id_of_index()
                        .rank(doc_id, start)?;
                    start = disk_multi_fm_index
                        .get_doc_id_of_index()
                        .select(doc_id, rank + 1)?
                        .unwrap_or(end);
                }
            }
        }
        Ok(Self {
            doc_id,
            k: start,
            end,
            multi_fm_index,
        })
    }
}

#[pymethods]
impl IterLocate {
    pub(super) fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub(super) fn __next__(mut slf: PyRefMut<Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if slf.k >= slf.end {
            return Ok(None);
        }
        let k = slf.k;
        let result = match &slf.multi_fm_index {
            MultiFMIndexEnum::InMemory(multi_fm_index) => {
                let (doc_id, offset) = py.detach(|| multi_fm_index.doc_offset(k))?;
                match slf.doc_id {
                    Some(_) => offset.into_pyobject(py)?.unbind().into(),
                    None => (doc_id, offset).into_pyobject(py)?.unbind().into(),
                }
            }
            MultiFMIndexEnum::OnDisk(disk_multi_fm_index) => {
                let (doc_id, offset) = py.detach(|| disk_multi_fm_index.doc_offset(k))?;
                match slf.doc_id {
                    Some(_) => offset.into_pyobject(py)?.unbind().into(),
                    None => (doc_id, offset).into_pyobject(py)?.unbind().into(),
                }
            }
        };
        slf.k = match slf.doc_id {
            Some(doc_id) => {
                let next_k = match &slf.multi_fm_index {
                    MultiFMIndexEnum::InMemory(multi_fm_index) => py.detach(|| {
                        let rank = multi_fm_index.get_doc_id_of_index().rank(doc_id, k + 1)?;
                        multi_fm_index
                            .get_doc_id_of_index()
                            .select(doc_id, rank + 1)
                    })?,
                    MultiFMIndexEnum::OnDisk(disk_multi_fm_index) => py.detach(|| {
                        let rank = disk_multi_fm_index
                            .get_doc_id_of_index()
                            .rank(doc_id, k + 1)?;
                        disk_multi_fm_index
                            .get_doc_id_of_index()
                            .select(doc_id, rank + 1)
                    })?,
                };
                next_k.unwrap_or(slf.end)
            }
            None => k + 1,
        };
        Ok(Some(result))
    }
}
