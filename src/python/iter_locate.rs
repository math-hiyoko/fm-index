use pyo3::{PyResult, prelude::*};

use super::fm_index::FMIndexEnum;
use crate::fm_index::traits::fm_index::FMIndexTrait;

#[pyclass]
pub(super) struct IterLocate {
    k: usize,
    end: usize,
    fm_index_enum: FMIndexEnum,
}

impl IterLocate {
    pub(super) fn new(pattern: &str, fm_index_enum: FMIndexEnum) -> PyResult<Self> {
        let (start, end) = match &fm_index_enum {
            FMIndexEnum::InMemory(fm_index) => fm_index.range_search(pattern)?,
            FMIndexEnum::OnDisk(disk_fm_index) => disk_fm_index.range_search(pattern)?,
        };
        Ok(Self {
            k: start,
            end,
            fm_index_enum,
        })
    }
}

#[pymethods]
impl IterLocate {
    pub(super) fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub(super) fn __next__(mut slf: PyRefMut<Self>, py: Python<'_>) -> PyResult<Option<usize>> {
        if slf.k >= slf.end {
            return Ok(None);
        }
        let k = slf.k;
        let result = match &slf.fm_index_enum {
            FMIndexEnum::InMemory(fm_index) => py.detach(|| fm_index.suffix_idx(k))?,
            FMIndexEnum::OnDisk(disk_fm_index) => py.detach(|| disk_fm_index.suffix_idx(k))?,
        };
        slf.k += 1;
        Ok(Some(result))
    }
}
