#![forbid(unsafe_code)]
mod fm_index;
mod python;
mod utils;

use pyo3::prelude::*;

use crate::python::{fm_index::PyFMIndex, multi_fm_index::PyMultiFMIndex};

#[pymodule(name = "fm_index")]
fn py_wavelet_matrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFMIndex>()?;
    m.add_class::<PyMultiFMIndex>()?;
    Ok(())
}
