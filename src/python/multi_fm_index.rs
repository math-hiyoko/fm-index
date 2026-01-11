use std::{char, collections, sync};

use pyo3::{
    PyResult,
    exceptions::{PyTypeError, PyUnicodeDecodeError},
    prelude::*,
    types::{IntoPyDict, PyDict, PyList, PySequence, PyString, PyStringData, PyStringMethods},
};
use rayon::prelude::*;

use crate::fm_index::multi_fm_index::MultiFMIndex;

enum IterLocateMultiFMIndexEnum {
    U8(sync::Arc<MultiFMIndex<u8>>),
    U16(sync::Arc<MultiFMIndex<u16>>),
    U32(sync::Arc<MultiFMIndex<u32>>),
}

#[pyclass]
struct IterLocate {
    k: usize,
    end: usize,
    multi_fm_index: sync::Arc<IterLocateMultiFMIndexEnum>,
}

impl IterLocate {
    fn new(multi_fm_index: IterLocateMultiFMIndexEnum, start: usize, end: usize) -> Self {
        IterLocate {
            k: start,
            end,
            multi_fm_index: sync::Arc::new(multi_fm_index),
        }
    }
}

#[pymethods]
impl IterLocate {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>, py: Python<'_>) -> PyResult<Option<(usize, usize)>> {
        if slf.k >= slf.end {
            return Ok(None);
        }
        let k = slf.k;
        let multi_fm_index = sync::Arc::clone(&slf.multi_fm_index);
        let result = py.detach(move || match &*multi_fm_index {
            IterLocateMultiFMIndexEnum::U8(multi_fm_index) => multi_fm_index.doc_offset(k),
            IterLocateMultiFMIndexEnum::U16(multi_fm_index) => multi_fm_index.doc_offset(k),
            IterLocateMultiFMIndexEnum::U32(multi_fm_index) => multi_fm_index.doc_offset(k),
        })?;
        slf.k += 1;
        Ok(Some(result))
    }
}

#[derive(Clone)]
enum MultiFMIndexEnum {
    U8(sync::Arc<MultiFMIndex<u8>>),
    U16(sync::Arc<MultiFMIndex<u16>>),
    U32(sync::Arc<MultiFMIndex<u32>>),
}

/// A multi-document FM-index for fast substring search across multiple strings.  
///
/// Internally, all strings are concatenated with separators and indexed as a single FM-index,  
/// while preserving the ability to map matches back to their original documents.  
/// Query processing across documents is internally parallelized where applicable,  
/// making multi-document search efficient in practice.  
///
/// ### Construction
/// #### Time / Space Complexity
/// - Time: `O(S log σ)`
/// - Space: `O(S log σ)`
///
/// where:
/// - `S` = total length of all indexed strings
/// - `σ` = size of the alphabet (2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.)
///
/// ```python
/// from fm_index import MultiFMIndex
///
/// mfm = MultiFMIndex(["abcabcabcabc", "xxabcabcxxabc", "abcababcabc"])
/// ```
#[derive(Clone)]
#[pyclass(name = "MultiFMIndex")]
pub(crate) struct PyMultiFMIndex {
    inner: MultiFMIndexEnum,
}

#[pymethods]
impl PyMultiFMIndex {
    /// Create a MultiFMIndex from the given list of strings.
    #[new]
    fn new(py: Python<'_>, data: &Bound<'_, PySequence>) -> PyResult<Self> {
        #[derive(PartialEq, PartialOrd, Eq, Ord)]
        enum StringData {
            Ucs1(Vec<u8>),
            Ucs2(Vec<u16>),
            Ucs4(Vec<u32>),
        }

        let data = data
            .try_iter()?
            .map(|item| {
                let item = item?;
                let item = item.cast::<PyString>().map_err(|_| {
                    PyTypeError::new_err("All elements in the sequence must be strings.")
                })?;
                match unsafe { item.data()? } {
                    PyStringData::Ucs1(data) => Ok(StringData::Ucs1(data.to_vec())),
                    PyStringData::Ucs2(data) => Ok(StringData::Ucs2(data.to_vec())),
                    PyStringData::Ucs4(data) => Ok(StringData::Ucs4(data.to_vec())),
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        py.detach(
            move || match data.par_iter().max().unwrap_or(&StringData::Ucs1(vec![])) {
                StringData::Ucs1(_) => {
                    let data = data
                        .into_par_iter()
                        .map(|item| match item {
                            StringData::Ucs1(data) => data,
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    let multi_multi_fm_index = MultiFMIndex::new(&data)?;
                    Ok(PyMultiFMIndex {
                        inner: MultiFMIndexEnum::U8(sync::Arc::new(multi_multi_fm_index)),
                    })
                }
                StringData::Ucs2(_) => {
                    let data = data
                        .into_par_iter()
                        .map(|item| match item {
                            StringData::Ucs1(data) => {
                                data.par_iter().map(|&c| c as u16).collect::<Vec<_>>()
                            }
                            StringData::Ucs2(data) => data,
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    let multi_multi_fm_index = MultiFMIndex::new(&data)?;
                    Ok(PyMultiFMIndex {
                        inner: MultiFMIndexEnum::U16(sync::Arc::new(multi_multi_fm_index)),
                    })
                }
                StringData::Ucs4(_) => {
                    let data = data
                        .into_par_iter()
                        .map(|item| match item {
                            StringData::Ucs1(data) => {
                                data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                            }
                            StringData::Ucs2(data) => {
                                data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                            }
                            StringData::Ucs4(data) => data,
                        })
                        .collect::<Vec<_>>();
                    let multi_multi_fm_index = MultiFMIndex::new(&data)?;
                    Ok(PyMultiFMIndex {
                        inner: MultiFMIndexEnum::U32(sync::Arc::new(multi_multi_fm_index)),
                    })
                }
            },
        )
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_multi_fm_index) => multi_multi_fm_index.len(),
            MultiFMIndexEnum::U16(multi_multi_fm_index) => multi_multi_fm_index.len(),
            MultiFMIndexEnum::U32(multi_multi_fm_index) => multi_multi_fm_index.len(),
        })
    }

    fn __contains__(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        self.contains(py, pattern)
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let result = py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .par_iter()
                    .map(|value| {
                        String::from_utf8(value.to_vec()).map_err(PyUnicodeDecodeError::new_err)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                PyResult::Ok(format!("MultiFMIndex({:?})", str_list))
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .par_iter()
                    .map(|value| String::from_utf16(value).map_err(PyUnicodeDecodeError::new_err))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(format!("MultiFMIndex({:?})", str_list))
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .par_iter()
                    .map(|value| {
                        Ok(value
                            .iter()
                            .map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
                            .collect::<String>())
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(format!("MultiFMIndex({:?})", str_list))
            }
        })?;
        Ok(PyString::new(py, &result).into())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        self.__str__(py)
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(move || Ok(self.clone()))
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__copy__(py)
    }

    /// Convert the index back into the original list of strings.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(S log σ)`
    /// - Space: `O(S)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.item()
    /// # ['abcabcabcabc', 'xxabcabcxxabc', 'abcababcabc']
    /// ```
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let str_list = py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .par_iter()
                    .map(|value| {
                        String::from_utf8(value.to_vec()).map_err(PyUnicodeDecodeError::new_err)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                PyResult::Ok(str_list)
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .par_iter()
                    .map(|value| String::from_utf16(value).map_err(PyUnicodeDecodeError::new_err))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(str_list)
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .par_iter()
                    .map(|value| {
                        Ok(value
                            .iter()
                            .map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
                            .collect::<String>())
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(str_list)
            }
        })?;
        Ok(PyList::new(py, str_list)?.unbind())
    }

    /// Check if the pattern exists as a full document in the index.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern| log σ)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.contains("abcabcabcabc")
    /// # True
    /// ```
    fn contains(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        let pattern = unsafe { pattern.data()? };
        py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => data,
                    _ => return Ok(false),
                };
                multi_fm_index.contains(pattern)
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u16).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(false),
                };
                multi_fm_index.contains(pattern)
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs4(data) => data,
                };
                multi_fm_index.contains(pattern)
            }
        })
    }

    /// Count total occurrences of a pattern across all documents.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern| log σ)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.count_all("abc")
    /// # 10
    /// ```
    fn count_all(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<usize> {
        let pattern = unsafe { pattern.data()? };
        py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => data,
                    _ => return Ok(0usize),
                };
                multi_fm_index.count_all(pattern)
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u16).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(0usize),
                };
                multi_fm_index.count_all(pattern)
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs4(data) => data,
                };
                multi_fm_index.count_all(pattern)
            }
        })
    }

    /// Count occurrences per document.  
    /// Returns {doc_index: count}.
    ///
    /// #### Complexity
    ///
    /// - Time: `O((|pattern| + |total_count|) log σ)`
    /// - Space: `O(|pattern| + |output|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.count("abc")
    /// # {0: 4, 1: 3, 2: 3}
    /// ```
    fn count(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<Py<PyDict>> {
        let pattern = unsafe { pattern.data()? };
        let count = py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => data,
                    _ => return Ok(collections::HashMap::new()),
                };
                multi_fm_index.count(pattern)
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u16).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(collections::HashMap::new()),
                };
                multi_fm_index.count(pattern)
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs4(data) => data,
                };
                multi_fm_index.count(pattern)
            }
        })?;
        Ok(count.into_py_dict(py)?.unbind())
    }

    /// Locate occurrences per document.  
    /// Internally, result enumeration and aggregation may be parallelized.  
    /// ⚠️ Order is not guaranteed.
    ///
    /// #### Complexity
    ///
    /// - Time: `O((|pattern| + |total_count|) log σ)`
    /// - Space: `O(|pattern| + |total_count|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.locate("abc")
    /// # {0: [9, 6, 3, 0], 1: [10, 2, 5], 2: [8, 0, 5]}
    /// ```
    fn locate(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<Py<PyDict>> {
        let pattern = unsafe { pattern.data()? };
        let locate = py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => data,
                    _ => return PyResult::Ok(collections::HashMap::new()),
                };
                Ok(multi_fm_index.locate(pattern)?)
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u16).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(collections::HashMap::new()),
                };
                Ok(multi_fm_index.locate(pattern)?)
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs4(data) => data,
                };
                Ok(multi_fm_index.locate(pattern)?)
            }
        })?;
        Ok(locate.into_py_dict(py)?.unbind())
    }

    /// Lazily locate all occurrences of the pattern across documents.
    ///
    /// Yields `(doc_index, position)` pairs without constructing
    /// an intermediate result dictionary.
    ///
    /// ⚠️ Order of yielded results is not guaranteed.
    ///
    /// ### Complexity
    ///
    /// - Time: `O(|pattern| log σ)` to initialize, then `O(log σ)` per yielded occurrence.
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// iter = mfm.iter_locate("abc")
    /// next(iter)
    /// # (2, 8)
    /// next(iter)
    /// # (1, 10)
    /// ...
    /// ```
    fn iter_locate(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<IterLocate> {
        let pattern = unsafe { pattern.data()? };
        py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => data,
                    _ => {
                        return PyResult::Ok(IterLocate::new(
                            IterLocateMultiFMIndexEnum::U8(multi_fm_index.clone()),
                            0,
                            0,
                        ));
                    }
                };
                let (start, end) = multi_fm_index.range_search(pattern)?;
                Ok(IterLocate::new(
                    IterLocateMultiFMIndexEnum::U8(multi_fm_index.clone()),
                    start,
                    end,
                ))
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u16).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => data,
                    _ => {
                        return Ok(IterLocate::new(
                            IterLocateMultiFMIndexEnum::U16(multi_fm_index.clone()),
                            0,
                            0,
                        ));
                    }
                };
                let (start, end) = multi_fm_index.range_search(pattern)?;
                Ok(IterLocate::new(
                    IterLocateMultiFMIndexEnum::U16(multi_fm_index.clone()),
                    start,
                    end,
                ))
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs4(data) => data,
                };
                let (start, end) = multi_fm_index.range_search(pattern)?;
                Ok(IterLocate::new(
                    IterLocateMultiFMIndexEnum::U32(multi_fm_index.clone()),
                    start,
                    end,
                ))
            }
        })
    }

    /// List document indices whose content starts with the prefix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|prefix| log σ)`
    /// - Space: `O(|prefix|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.startswith("abc")
    /// # [2, 0]
    /// ```
    fn startswith(&self, py: Python<'_>, prefix: &Bound<'_, PyString>) -> PyResult<Py<PyList>> {
        let prefix = unsafe { prefix.data()? };
        let result = py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let prefix = match prefix {
                    PyStringData::Ucs1(data) => data,
                    _ => return Ok(vec![]),
                };
                multi_fm_index.starts_with(prefix)
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let prefix = match prefix {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u16).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(vec![]),
                };
                multi_fm_index.starts_with(prefix)
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let prefix = match prefix {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs4(data) => data,
                };
                multi_fm_index.starts_with(prefix)
            }
        })?;
        Ok(PyList::new(py, result)?.unbind())
    }

    /// List document indices whose content ends with the suffix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|suffix| log σ)`
    /// - Space: `O(|suffix|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.endswith("abc")
    /// # [2, 1, 0]
    /// ```
    fn endswith(&self, py: Python<'_>, suffix: &Bound<'_, PyString>) -> PyResult<Py<PyList>> {
        let suffix = unsafe { suffix.data()? };
        let result = py.detach(move || match &self.inner {
            MultiFMIndexEnum::U8(multi_fm_index) => {
                let suffix = match suffix {
                    PyStringData::Ucs1(data) => data,
                    _ => return Ok(vec![]),
                };
                multi_fm_index.ends_with(suffix)
            }
            MultiFMIndexEnum::U16(multi_fm_index) => {
                let suffix = match suffix {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u16).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(vec![]),
                };
                multi_fm_index.ends_with(suffix)
            }
            MultiFMIndexEnum::U32(multi_fm_index) => {
                let suffix = match suffix {
                    PyStringData::Ucs1(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs2(data) => {
                        &data.par_iter().map(|&c| c as u32).collect::<Vec<_>>()
                    }
                    PyStringData::Ucs4(data) => data,
                };
                multi_fm_index.ends_with(suffix)
            }
        })?;
        Ok(PyList::new(py, result)?.unbind())
    }
}

#[cfg(test)]
mod tests {
    use pyo3::Python;

    use super::*;

    #[test]
    fn test_multi_fm_index_empty_list() {
        Python::initialize();

        Python::attach(|py| {
            let values = Vec::<String>::new();
            let pylist = PyList::new(py, &values).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let multi_fm_index = PyMultiFMIndex::new(py, pysequence).unwrap();

            assert_eq!(multi_fm_index.__len__(py).unwrap(), 0);
            assert!(multi_fm_index.__copy__(py).is_ok());
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, ""))
                    .unwrap()
            );
            assert_eq!(
                multi_fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                format!("MultiFMIndex({:?})", values)
            );
            assert_eq!(
                multi_fm_index
                    .item(py)
                    .unwrap()
                    .extract::<Vec<String>>(py)
                    .unwrap(),
                values
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, ""))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "a"))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
        });
    }

    #[test]
    fn test_multi_fm_index_empties() {
        Python::initialize();

        Python::attach(|py| {
            let values = ["", "", ""];
            let pylist = PyList::new(py, values).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let multi_fm_index = PyMultiFMIndex::new(py, pysequence).unwrap();

            assert_eq!(multi_fm_index.__len__(py).unwrap(), 3);
            assert!(multi_fm_index.__copy__(py).is_ok());
            assert_eq!(
                multi_fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                format!("MultiFMIndex({:?})", values)
            );
            assert_eq!(
                multi_fm_index
                    .item(py)
                    .unwrap()
                    .extract::<Vec<String>>(py)
                    .unwrap(),
                values
            );
            assert!(
                multi_fm_index
                    .__contains__(py, &PyString::new(py, ""))
                    .unwrap()
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, ""))
                    .unwrap(),
                3
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "a"))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::from([(0, 1), (1, 1), (2, 1)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::from([(0, [0]), (1, [0]), (2, [0])])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 1, 0]
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 1, 0]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
        });
    }

    #[test]
    fn test_multi_fm_index_ucs1() {
        Python::initialize();

        Python::attach(|py| {
            let values = ["abcabcabcabc", "xxabcabcxxabc", "abcababcabc"];
            let pylist = PyList::new(py, values).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let multi_fm_index = PyMultiFMIndex::new(py, pysequence).unwrap();

            assert_eq!(multi_fm_index.__len__(py).unwrap(), 3);
            assert!(multi_fm_index.__copy__(py).is_ok());
            assert_eq!(
                multi_fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                format!("MultiFMIndex({:?})", values)
            );
            assert_eq!(
                multi_fm_index
                    .item(py)
                    .unwrap()
                    .extract::<Vec<String>>(py)
                    .unwrap(),
                values
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "abc"))
                    .unwrap()
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "あいう"))
                    .unwrap()
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
            );
            assert!(
                multi_fm_index
                    .__contains__(py, &PyString::new(py, "abcabcabcabc"))
                    .unwrap()
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, ""))
                    .unwrap(),
                39
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "abc"))
                    .unwrap(),
                10
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "あいう"))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "😀😃😀"))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 13), (1, 14), (2, 12)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 4), (1, 3), (2, 3)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![12, 9, 6, 3, 0, 10, 7, 4, 1, 11, 8, 5, 2]),
                    (1, vec![13, 10, 2, 5, 11, 3, 6, 12, 4, 7, 9, 1, 8, 0]),
                    (2, vec![11, 3, 8, 0, 5, 4, 9, 1, 6, 10, 2, 7])
                ])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![9, 6, 3, 0]),
                    (1, vec![10, 2, 5]),
                    (2, vec![8, 0, 5])
                ])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            let iter_locate = multi_fm_index
                .iter_locate(py, &PyString::new(py, "abc"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some((2, 8))
            );
            assert!(
                multi_fm_index
                    .iter_locate(py, &PyString::new(py, "あいう"))
                    .is_ok()
            );
            assert!(
                multi_fm_index
                    .iter_locate(py, &PyString::new(py, "😀😃😀"))
                    .is_ok()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0, 1]
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0]
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 1, 0]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "xabc"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [1]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
        });
    }

    #[test]
    fn test_multi_fm_index_ucs2() {
        Python::initialize();

        Python::attach(|py| {
            let values = ["あいうあいうあいう", "xxあいうあいうxx", "あいうあいあいう"];
            let pylist = PyList::new(py, values).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let multi_fm_index = PyMultiFMIndex::new(py, pysequence).unwrap();

            assert_eq!(multi_fm_index.__len__(py).unwrap(), 3);
            assert!(multi_fm_index.__copy__(py).is_ok());
            assert_eq!(
                multi_fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                format!("MultiFMIndex({:?})", values)
            );
            assert_eq!(
                multi_fm_index
                    .item(py)
                    .unwrap()
                    .extract::<Vec<String>>(py)
                    .unwrap(),
                values
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "issi"))
                    .unwrap()
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "あいう"))
                    .unwrap()
            );
            assert!(
                multi_fm_index
                    .__contains__(py, &PyString::new(py, "あいうあいうあいう"))
                    .unwrap()
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, ""))
                    .unwrap(),
                30
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "abc"))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "あいう"))
                    .unwrap(),
                7
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "😀😃😀"))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 10), (1, 11), (2, 9)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 3), (1, 2), (2, 2)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![9, 6, 3, 0, 7, 4, 1, 8, 5, 2]),
                    (1, vec![10, 9, 8, 0, 1, 5, 2, 6, 3, 7, 4]),
                    (2, vec![8, 3, 5, 0, 4, 6, 1, 7, 2])
                ])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![6, 3, 0]),
                    (1, vec![5, 2]),
                    (2, vec![5, 0])
                ])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            let iter_locate = multi_fm_index
                .iter_locate(py, &PyString::new(py, "あいう"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some((2, 5))
            );
            assert!(
                multi_fm_index
                    .iter_locate(py, &PyString::new(py, "abc"))
                    .is_ok()
            );
            assert!(
                multi_fm_index
                    .iter_locate(py, &PyString::new(py, "😀😃😀"))
                    .is_ok()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [1, 2, 0]
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0]
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0, 1]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
        });
    }

    #[test]
    fn test_multi_fm_index_ucs4() {
        Python::initialize();

        Python::attach(|py| {
            let values = ["😀😃😀😃😀😃", "xx😀😃😀😃xx", "😀😃😀😀😃"];
            let pylist = PyList::new(py, values).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let multi_fm_index = PyMultiFMIndex::new(py, pysequence).unwrap();

            assert_eq!(multi_fm_index.__len__(py).unwrap(), 3);
            assert!(multi_fm_index.__copy__(py).is_ok());
            assert_eq!(
                multi_fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                format!("MultiFMIndex({:?})", values)
            );
            assert_eq!(
                multi_fm_index
                    .item(py)
                    .unwrap()
                    .extract::<Vec<String>>(py)
                    .unwrap(),
                values
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "abc"))
                    .unwrap()
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "あいう"))
                    .unwrap()
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "😀😃"))
                    .unwrap()
            );
            assert!(
                multi_fm_index
                    .__contains__(py, &PyString::new(py, "😀😃😀😃😀😃"))
                    .unwrap()
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, ""))
                    .unwrap(),
                22
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "abc"))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "あいう"))
                    .unwrap(),
                0
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "😀😃😀"))
                    .unwrap(),
                4
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 7), (1, 9), (2, 6)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 2), (1, 1), (2, 1)])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![6, 4, 2, 0, 5, 3, 1]),
                    (1, vec![8, 7, 6, 0, 1, 4, 2, 5, 3]),
                    (2, vec![5, 2, 3, 0, 4, 1])
                ])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![2, 0]),
                    (1, vec![2]),
                    (2, vec![0])
                ])
            );
            let iter_locate = multi_fm_index
                .iter_locate(py, &PyString::new(py, "😀😃😀"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some((2, 0))
            );
            assert!(
                multi_fm_index
                    .iter_locate(py, &PyString::new(py, "abc"))
                    .is_ok()
            );
            assert!(
                multi_fm_index
                    .iter_locate(py, &PyString::new(py, "あいう"))
                    .is_ok()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [1, 2, 0]
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "😀😃😀"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0, 1]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "abc"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "あいう"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "😀😃"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0]
            );
        });
    }

    #[test]
    fn test_multi_fm_index_zwj() {
        Python::initialize();

        Python::attach(|py| {
            let values = ["👨‍👩‍👧‍👦👨‍👩‍👧‍👦", "xx👨‍👩‍👧‍👦xx", "👨‍👩‍👧‍👦👨‍👧"];
            let pylist = PyList::new(py, values).unwrap();
            let pysequence = pylist.cast::<PySequence>().unwrap();
            let multi_fm_index = PyMultiFMIndex::new(py, pysequence).unwrap();

            assert_eq!(multi_fm_index.__len__(py).unwrap(), 3);
            assert!(multi_fm_index.__copy__(py).is_ok());
            assert_eq!(
                multi_fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                format!("MultiFMIndex({:?})", values)
            );
            assert_eq!(
                multi_fm_index
                    .item(py)
                    .unwrap()
                    .extract::<Vec<String>>(py)
                    .unwrap(),
                values
            );
            assert!(
                !multi_fm_index
                    .__contains__(py, &PyString::new(py, "👨‍👩‍👧‍👦"))
                    .unwrap()
            );
            assert!(
                multi_fm_index
                    .__contains__(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦"))
                    .unwrap()
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, ""))
                    .unwrap(),
                38
            );
            assert_eq!(
                multi_fm_index
                    .count_all(py, &PyString::new(py, "👨‍👩‍👧‍👦"))
                    .unwrap(),
                4
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 15), (1, 12), (2, 11)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "👨‍👩‍👧‍👦"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 2), (1, 1), (2, 1)])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![14, 12, 5, 10, 3, 8, 1, 13, 6, 11, 4, 7, 0, 9, 2]),
                    (1, vec![11, 10, 9, 0, 1, 7, 5, 3, 8, 6, 2, 4]),
                    (2, vec![10, 5, 8, 3, 1, 6, 9, 4, 7, 0, 2])
                ])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "👨‍👩‍👧‍👦"))
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![7, 0]),
                    (1, vec![2]),
                    (2, vec![0])
                ])
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [1, 2, 0]
            );
            assert_eq!(
                multi_fm_index
                    .startswith(py, &PyString::new(py, "👨‍👩‍👧‍👦"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [2, 0, 1]
            );
            assert_eq!(
                multi_fm_index
                    .endswith(py, &PyString::new(py, "👨‍👩‍👧‍👦"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [0]
            );
        });
    }
}
