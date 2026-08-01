use std::{iter, sync};

use bytemuck::cast_slice_mut;
use memmap2::MmapMut;
use pyo3::{
    PyResult,
    exceptions::{PyTypeError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyBytesMethods, PyList, PyString, PyStringMethods},
};
use tempfile::tempfile;

use super::iter_locate::IterLocate;
use crate::fm_index::{
    fm_index::{disk_fm_index::DiskFMIndex, fm_index::FMIndex},
    traits::fm_index::FMIndexTrait,
};

pub(super) enum FMIndexEnum {
    InMemory(sync::Arc<FMIndex>),
    OnDisk(sync::Arc<DiskFMIndex>),
}

/// An FM-index for efficient full-text search on a single string.
///
/// The FM-index is a compressed text index based on the Burrows–Wheeler Transform (BWT).  
/// It supports fast substring queries whose runtime depends only on the pattern length,  
/// not on the size of the indexed text.  
/// Internally, several independent stages of index construction and query processing  
/// are optimized using data-parallel execution.  
///
/// ### Construction
/// #### Time / Space Complexity
/// - Time: `O(N)`
/// - Space: `O(N)`
///
/// where:
/// - `N` = length of the indexed string
///
/// ```python
/// from fm_index import FMIndex
///
/// fm = FMIndex("mississippi", on_disk=False)  # in-memory
/// ```
///
/// ### Serialization
/// FMIndex supports Python's pickle protocol for efficient persistence:
///
/// ```python
/// import pickle
///
/// # Save index (available only for in-memory FMIndex)
/// with open("index.pkl", "wb") as f:
///     pickle.dump(fm, f)
///
/// # Load index (available only for in-memory FMIndex)
/// with open("index.pkl", "rb") as f:
///     fm = pickle.load(f)
/// ```
#[pyclass(name = "FMIndex", skip_from_py_object)]
pub(crate) struct PyFMIndex {
    inner: FMIndexEnum,
}

#[pymethods]
impl PyFMIndex {
    /// Create a FM-Index from the given string.
    #[new]
    #[pyo3(signature = (data, on_disk=false))]
    fn new(py: Python<'_>, data: &Bound<'_, PyString>, on_disk: bool) -> PyResult<Self> {
        let data = data.to_str()?;
        let inner = py.detach(move || match on_disk {
            true => {
                let data_file = tempfile().map_err(PyRuntimeError::new_err)?;
                data_file
                    .set_len(((data.chars().count() + 1) * std::mem::size_of::<u32>()) as u64)
                    .map_err(PyRuntimeError::new_err)?;
                #[allow(unsafe_code)]
                let mut data_mmap =
                    unsafe { MmapMut::map_mut(&data_file).map_err(PyRuntimeError::new_err)? };
                let data_slice = cast_slice_mut::<u8, u32>(&mut data_mmap[..]);
                for (i, c) in data.chars().enumerate() {
                    data_slice[i] = c as u32 + 1;
                }
                data_slice[data.chars().count()] = 0; // null terminator
                let data_mmap = data_mmap
                    .make_read_only()
                    .map_err(PyRuntimeError::new_err)?;
                PyResult::Ok(FMIndexEnum::OnDisk(sync::Arc::new(DiskFMIndex::new(
                    data_mmap,
                )?)))
            }
            false => {
                let data = data
                    .chars()
                    .map(|c| c as u32 + 1)
                    .chain(iter::once(0)) // null terminator
                    .collect::<Vec<_>>();
                PyResult::Ok(FMIndexEnum::InMemory(sync::Arc::new(FMIndex::new(data)?)))
            }
        })?;
        Ok(PyFMIndex { inner })
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => fm_index.len(),
            FMIndexEnum::OnDisk(disk_fm_index) => disk_fm_index.len(),
        })
    }

    fn __contains__(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        let pattern = pattern.to_str()?;
        py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => fm_index.contains(pattern),
            FMIndexEnum::OnDisk(disk_fm_index) => disk_fm_index.contains(pattern),
        })
    }

    fn __str__(&self, py: Python<'_>) -> Py<PyString> {
        let result = py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => {
                let len = fm_index.len();
                let max_bit = fm_index.max_bit();
                format!("FMIndex(len={}, max_bit={}, on_disk=False)", len, max_bit)
            }
            FMIndexEnum::OnDisk(disk_fm_index) => {
                let len = disk_fm_index.len();
                let max_bit = disk_fm_index.max_bit();
                format!("FMIndex(len={}, max_bit={}, on_disk=True)", len, max_bit)
            }
        });
        PyString::new(py, &result).into()
    }

    fn __repr__(&self, py: Python<'_>) -> Py<PyString> {
        self.__str__(py)
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => {
                let fm_index_clone = fm_index.clone();
                PyResult::Ok(PyFMIndex {
                    inner: FMIndexEnum::InMemory(fm_index_clone),
                })
            }
            FMIndexEnum::OnDisk(disk_fm_index) => {
                let disk_fm_index_clone = disk_fm_index.clone();
                PyResult::Ok(PyFMIndex {
                    inner: FMIndexEnum::OnDisk(disk_fm_index_clone),
                })
            }
        })
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => Ok(PyFMIndex {
                inner: FMIndexEnum::InMemory(sync::Arc::new(fm_index.as_ref().clone())),
            }),
            FMIndexEnum::OnDisk(disk_fm_index) => {
                let disk_fm_index_clone = disk_fm_index.try_clone()?;
                PyResult::Ok(PyFMIndex {
                    inner: FMIndexEnum::OnDisk(sync::Arc::new(disk_fm_index_clone)),
                })
            }
        })
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        match &self.inner {
            FMIndexEnum::InMemory(_) => {
                // Return (class, args, state) where:
                // - class: the class to instantiate
                // - args: arguments for __new__ (empty string for us)
                // - state: will be passed to __setstate__
                let cls = py.import("fm_index")?.getattr("FMIndex")?.into();
                let args = (PyString::new(py, ""),)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind();
                let state: Py<PyAny> = self.__getstate__(py)?.into();
                Ok((cls, args, state))
            }
            FMIndexEnum::OnDisk(_) => Err(PyTypeError::new_err(
                "__reduce__ is not supported for on-disk FMIndex",
            )),
        }
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        match &self.inner {
            FMIndexEnum::InMemory(fm_index) => {
                let serialized = postcard::to_allocvec(fm_index.as_ref()).map_err(|error| {
                    PyValueError::new_err(format!("Failed to serialize: {}", error))
                })?;
                Ok(PyBytes::new(py, &serialized).into())
            }
            FMIndexEnum::OnDisk(_) => Err(PyTypeError::new_err(
                "__getstate__ is not supported for on-disk FMIndex",
            )),
        }
    }

    fn __setstate__(&mut self, _py: Python<'_>, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        match &self.inner {
            FMIndexEnum::InMemory(_) => {
                let bytes = state.as_bytes();
                let inner: FMIndex = postcard::from_bytes(bytes).map_err(|error| {
                    PyValueError::new_err(format!("Failed to deserialize: {}", error))
                })?;
                self.inner = FMIndexEnum::InMemory(sync::Arc::new(inner));
                Ok(())
            }
            FMIndexEnum::OnDisk(_) => Err(PyTypeError::new_err(
                "__setstate__ is not supported for on-disk FMIndex",
            )),
        }
    }

    /// Convert the FMIndex back into the original string.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(N)`
    /// - Space: `O(N)`
    ///
    /// #### Examples
    /// ```python
    /// fm.item()
    /// # 'mississippi'
    /// ```
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let str = py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => fm_index.value(),
            FMIndexEnum::OnDisk(disk_fm_index) => disk_fm_index.value(),
        })?;
        Ok(PyString::new(py, &str).into())
    }

    /// Check whether the indexed string contains the given pattern.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern|)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.contains("issi")
    /// # True
    /// ```
    fn contains(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        self.__contains__(py, pattern)
    }

    /// Count how many times a pattern appears in the indexed string.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern|)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.count("issi")
    /// # 2
    /// ```
    fn count(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<usize> {
        let pattern = pattern.to_str()?;
        py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => fm_index.count(pattern),
            FMIndexEnum::OnDisk(disk_fm_index) => disk_fm_index.count(pattern),
        })
    }

    /// Locate all starting positions of the pattern in the indexed string.  
    /// This operation may internally leverage parallel execution to efficiently  
    /// enumerate large result sets.  
    /// ⚠️ Order of returned positions is not guaranteed.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern| + |count|)`
    /// - Space: `O(|pattern| + |count|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.locate("issi")
    /// # [4, 1]
    /// ```
    fn locate(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<Py<PyList>> {
        let pattern = pattern.to_str()?;
        let locate = py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => fm_index.locate(pattern),
            FMIndexEnum::OnDisk(disk_fm_index) => disk_fm_index.locate(pattern),
        })?;
        Ok(PyList::new(py, &locate)?.unbind())
    }

    /// Lazily locate all starting positions of the pattern in the indexed string.
    ///
    /// This method yields the same positions as `locate`, but returns them
    /// one by one as an iterator instead of allocating a list.
    ///
    /// ⚠️ Order of yielded positions is not guaranteed.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern|)` for initialization, `O(1)` per yielded position
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// iter = fm.iter_locate("issi")
    /// next(iter)
    /// # 4
    /// next(iter)
    /// # 1
    /// ```
    fn iter_locate(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<IterLocate> {
        let pattern = pattern.to_str()?;
        let fm_index = match &self.inner {
            FMIndexEnum::InMemory(fm_index) => FMIndexEnum::InMemory(fm_index.clone()),
            FMIndexEnum::OnDisk(disk_fm_index) => FMIndexEnum::OnDisk(disk_fm_index.clone()),
        };
        py.detach(move || IterLocate::new(pattern, fm_index))
    }

    /// Check if the indexed string starts with the given prefix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|prefix|)`
    /// - Space: `O(|prefix|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.startswith("mi")
    /// # True
    /// ```
    fn startswith(&self, py: Python<'_>, prefix: &Bound<'_, PyString>) -> PyResult<bool> {
        let prefix = prefix.to_str()?;
        py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => fm_index.starts_with(prefix),
            FMIndexEnum::OnDisk(disk_fm_index) => disk_fm_index.starts_with(prefix),
        })
    }

    /// Check if the indexed string ends with the given suffix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|suffix|)`
    /// - Space: `O(|suffix|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.endswith("pi")
    /// # True
    /// ```
    fn endswith(&self, py: Python<'_>, suffix: &Bound<'_, PyString>) -> PyResult<bool> {
        let suffix = suffix.to_str()?;
        py.detach(|| match &self.inner {
            FMIndexEnum::InMemory(fm_index) => fm_index.ends_with(suffix),
            FMIndexEnum::OnDisk(disk_fm_index) => disk_fm_index.ends_with(suffix),
        })
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{Python, types::PyString};

    use super::*;

    #[test]
    fn test_fm_index_empty() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index = PyFMIndex::new(py, &PyString::new(py, ""), false).unwrap();

            assert_eq!(fm_index.__len__(py), 0);
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=0, max_bit=0, on_disk=False)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                ""
            );
            assert!(fm_index.__contains__(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.__contains__(py, &PyString::new(py, "a")).unwrap());
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 1);
            assert_eq!(fm_index.count(py, &PyString::new(py, "a")).unwrap(), 0);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [0]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "a")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "a")).unwrap());
        });
    }

        #[test]
    fn test_disk_fm_index_empty() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index = PyFMIndex::new(py, &PyString::new(py, ""), true).unwrap();

            assert_eq!(fm_index.__len__(py), 0);
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=0, max_bit=0, on_disk=True)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                ""
            );
            assert!(fm_index.__contains__(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.__contains__(py, &PyString::new(py, "a")).unwrap());
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 1);
            assert_eq!(fm_index.count(py, &PyString::new(py, "a")).unwrap(), 0);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [0]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "a"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "a")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "a")).unwrap());
        });
    }

    #[test]
    fn test_fm_index_ucs1() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "mississippi"), false).unwrap();

            assert_eq!(fm_index.__len__(py), 11);
            assert!(
                fm_index
                    .__contains__(py, &PyString::new(py, "issi"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "にわ"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=11, max_bit=7, on_disk=False)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "mississippi"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 12);
            assert_eq!(fm_index.count(py, &PyString::new(py, "issi")).unwrap(), 2);
            assert_eq!(fm_index.count(py, &PyString::new(py, "にわ")).unwrap(), 0);
            assert_eq!(fm_index.count(py, &PyString::new(py, "🐉🔥🌊")).unwrap(), 0);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [11, 10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "issi"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [4, 1]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "にわ"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            );
            let iter_locate = fm_index
                .iter_locate(py, &PyString::new(py, "issi"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some(4)
            );
            assert!(fm_index.iter_locate(py, &PyString::new(py, "にわ")).is_ok());
            assert!(
                fm_index
                    .iter_locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .is_ok()
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.startswith(py, &PyString::new(py, "miss")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "にわ")).unwrap());
            assert!(
                !fm_index
                    .startswith(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .startswith(py, &PyString::new(py, "いっぴ"))
                    .unwrap()
            );
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "ippi")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "にわ")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "🐉🔥🌊")).unwrap());
        });
    }

        #[test]
    fn test_disk_fm_index_ucs1() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "mississippi"), true).unwrap();

            assert_eq!(fm_index.__len__(py), 11);
            assert!(
                fm_index
                    .__contains__(py, &PyString::new(py, "issi"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "にわ"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=11, max_bit=7, on_disk=True)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "mississippi"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 12);
            assert_eq!(fm_index.count(py, &PyString::new(py, "issi")).unwrap(), 2);
            assert_eq!(fm_index.count(py, &PyString::new(py, "にわ")).unwrap(), 0);
            assert_eq!(fm_index.count(py, &PyString::new(py, "🐉🔥🌊")).unwrap(), 0);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [11, 10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "issi"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [4, 1]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "にわ"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            );
            let iter_locate = fm_index
                .iter_locate(py, &PyString::new(py, "issi"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some(4)
            );
            assert!(fm_index.iter_locate(py, &PyString::new(py, "にわ")).is_ok());
            assert!(
                fm_index
                    .iter_locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .is_ok()
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.startswith(py, &PyString::new(py, "miss")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "にわ")).unwrap());
            assert!(
                !fm_index
                    .startswith(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .startswith(py, &PyString::new(py, "いっぴ"))
                    .unwrap()
            );
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "ippi")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "にわ")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "🐉🔥🌊")).unwrap());
        });
    }

    #[test]
    fn test_fm_index_ucs2() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index =
                PyFMIndex::new(py, &PyString::new(py, "にわにはにわにわとりがいる"), false)
                    .unwrap();

            assert_eq!(fm_index.__len__(py), 13);
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=13, max_bit=14, on_disk=False)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "issi"))
                    .unwrap()
            );
            assert!(
                fm_index
                    .__contains__(py, &PyString::new(py, "にわ"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "にわにはにわにわとりがいる"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 14);
            assert_eq!(fm_index.count(py, &PyString::new(py, "issi")).unwrap(), 0);
            assert_eq!(fm_index.count(py, &PyString::new(py, "にわ")).unwrap(), 3);
            assert_eq!(fm_index.count(py, &PyString::new(py, "🐉🔥🌊")).unwrap(), 0);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [13, 11, 10, 8, 2, 6, 0, 4, 3, 9, 12, 7, 1, 5]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "issi"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "にわ"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [6, 0, 4]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            let iter_locate = fm_index
                .iter_locate(py, &PyString::new(py, "にわ"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some(6)
            );
            assert!(fm_index.iter_locate(py, &PyString::new(py, "issi")).is_ok());
            assert!(
                fm_index
                    .iter_locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .is_ok()
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "issi")).unwrap());
            assert!(
                fm_index
                    .startswith(py, &PyString::new(py, "にわに"))
                    .unwrap()
            );
            assert!(!fm_index.startswith(py, &PyString::new(py, "🐓")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "issi")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "がいる")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "🕊️")).unwrap());
        });
    }

        #[test]
    fn test_disk_fm_index_ucs2() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index =
                PyFMIndex::new(py, &PyString::new(py, "にわにはにわにわとりがいる"), true)
                    .unwrap();

            assert_eq!(fm_index.__len__(py), 13);
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=13, max_bit=14, on_disk=True)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "issi"))
                    .unwrap()
            );
            assert!(
                fm_index
                    .__contains__(py, &PyString::new(py, "にわ"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "にわにはにわにわとりがいる"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 14);
            assert_eq!(fm_index.count(py, &PyString::new(py, "issi")).unwrap(), 0);
            assert_eq!(fm_index.count(py, &PyString::new(py, "にわ")).unwrap(), 3);
            assert_eq!(fm_index.count(py, &PyString::new(py, "🐉🔥🌊")).unwrap(), 0);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [13, 11, 10, 8, 2, 6, 0, 4, 3, 9, 12, 7, 1, 5]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "issi"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "にわ"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [6, 0, 4]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new()
            );
            let iter_locate = fm_index
                .iter_locate(py, &PyString::new(py, "にわ"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some(6)
            );
            assert!(fm_index.iter_locate(py, &PyString::new(py, "issi")).is_ok());
            assert!(
                fm_index
                    .iter_locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .is_ok()
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "issi")).unwrap());
            assert!(
                fm_index
                    .startswith(py, &PyString::new(py, "にわに"))
                    .unwrap()
            );
            assert!(!fm_index.startswith(py, &PyString::new(py, "🐓")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "issi")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "がいる")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "🕊️")).unwrap());
        });
    }

    #[test]
    fn test_fm_index_ucs4() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index =
                PyFMIndex::new(py, &PyString::new(py, "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊"), false)
                    .unwrap();

            assert_eq!(fm_index.__len__(py), 15);
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "issi"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "にわ"))
                    .unwrap()
            );
            assert!(
                fm_index
                    .__contains__(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=15, max_bit=17, on_disk=False)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 16);
            assert_eq!(fm_index.count(py, &PyString::new(py, "issi")).unwrap(), 0);
            assert_eq!(fm_index.count(py, &PyString::new(py, "にわ")).unwrap(), 0);
            assert_eq!(fm_index.count(py, &PyString::new(py, "🐉🔥🌊")).unwrap(), 3);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [15, 9, 5, 10, 11, 14, 8, 3, 4, 0, 12, 6, 1, 13, 7, 2]
            ); // "⚔️" counts as 2 letters
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "issi"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "にわ"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [12, 6, 1],
            ); // "⚔️" counts as 2 letters
            let iter_locate = fm_index
                .iter_locate(py, &PyString::new(py, "🐉🔥"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some(12)
            );
            assert!(fm_index.iter_locate(py, &PyString::new(py, "issi")).is_ok());
            assert!(fm_index.iter_locate(py, &PyString::new(py, "にわ")).is_ok());
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "issi")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "にわ")).unwrap());
            assert!(
                fm_index
                    .startswith(py, &PyString::new(py, "🏰🐉🔥"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .startswith(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "issi")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "にわ")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "🐉🔥🌊")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "⚔️🐉🔥")).unwrap());
        });
    }

        #[test]
    fn test_disk_fm_index_ucs4() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index =
                PyFMIndex::new(py, &PyString::new(py, "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊"), true)
                    .unwrap();

            assert_eq!(fm_index.__len__(py), 15);
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "issi"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .__contains__(py, &PyString::new(py, "にわ"))
                    .unwrap()
            );
            assert!(
                fm_index
                    .__contains__(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=15, max_bit=17, on_disk=True)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 16);
            assert_eq!(fm_index.count(py, &PyString::new(py, "issi")).unwrap(), 0);
            assert_eq!(fm_index.count(py, &PyString::new(py, "にわ")).unwrap(), 0);
            assert_eq!(fm_index.count(py, &PyString::new(py, "🐉🔥🌊")).unwrap(), 3);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [15, 9, 5, 10, 11, 14, 8, 3, 4, 0, 12, 6, 1, 13, 7, 2]
            ); // "⚔️" counts as 2 letters
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "issi"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "にわ"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [12, 6, 1],
            ); // "⚔️" counts as 2 letters
            let iter_locate = fm_index
                .iter_locate(py, &PyString::new(py, "🐉🔥"))
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some(12)
            );
            assert!(fm_index.iter_locate(py, &PyString::new(py, "issi")).is_ok());
            assert!(fm_index.iter_locate(py, &PyString::new(py, "にわ")).is_ok());
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "issi")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "にわ")).unwrap());
            assert!(
                fm_index
                    .startswith(py, &PyString::new(py, "🏰🐉🔥"))
                    .unwrap()
            );
            assert!(
                !fm_index
                    .startswith(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
            );
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "issi")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "にわ")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "🐉🔥🌊")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "⚔️🐉🔥")).unwrap());
        });
    }

    #[test]
    fn test_fm_index_zwj() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦👨‍👧"), false).unwrap();

            assert_eq!(fm_index.__len__(py), 35);
            assert!(fm_index.__contains__(py, &PyString::new(py, "👨‍👩‍👧‍👦")).unwrap());
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=35, max_bit=17, on_disk=False)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "👨‍👩‍👧‍👦👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦👨‍👧"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 36);
            assert_eq!(fm_index.count(py, &PyString::new(py, "👨‍👩‍👧‍👦")).unwrap(), 4);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [
                    35, 14, 23, 15, 24, 12, 21, 30, 5, 33, 10, 19, 28, 3, 8, 17, 26, 1, 13, 22, 31,
                    6, 34, 11, 20, 29, 4, 32, 7, 16, 25, 0, 9, 18, 27, 2
                ]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "👨‍👩‍👧‍👦"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [7, 16, 25, 0]
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.startswith(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👧")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👧")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦")).unwrap());
        });
    }

        #[test]
    fn test_disk_fm_index_zwj() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦👨‍👧"), true).unwrap();

            assert_eq!(fm_index.__len__(py), 35);
            assert!(fm_index.__contains__(py, &PyString::new(py, "👨‍👩‍👧‍👦")).unwrap());
            assert_eq!(
                fm_index.__repr__(py).extract::<String>(py).unwrap(),
                "FMIndex(len=35, max_bit=17, on_disk=True)"
            );
            assert!(fm_index.__copy__(py).is_ok());
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "👨‍👩‍👧‍👦👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦👨‍👧"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 36);
            assert_eq!(fm_index.count(py, &PyString::new(py, "👨‍👩‍👧‍👦")).unwrap(), 4);
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, ""))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [
                    35, 14, 23, 15, 24, 12, 21, 30, 5, 33, 10, 19, 28, 3, 8, 17, 26, 1, 13, 22, 31,
                    6, 34, 11, 20, 29, 4, 32, 7, 16, 25, 0, 9, 18, 27, 2
                ]
            );
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "👨‍👩‍👧‍👦"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [7, 16, 25, 0]
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.startswith(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦")).unwrap());
            assert!(!fm_index.startswith(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👧")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👧")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦")).unwrap());
        });
    }
}
