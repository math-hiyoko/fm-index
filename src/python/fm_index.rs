use std::{collections, iter, sync};

use pyo3::{
    PyResult,
    exceptions::PyUnicodeDecodeError,
    prelude::*,
    types::{PyList, PyString, PyStringMethods},
};

use crate::fm_index::fm_index::FMIndex;

#[pyclass]
struct IterLocate {
    k: usize,
    end: usize,
    fm_index: sync::Arc<FMIndex>,
    char_indices: sync::Arc<collections::HashMap<usize, usize>>,
}

#[pymethods]
impl IterLocate {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>, py: Python<'_>) -> PyResult<Option<usize>> {
        let k = slf.k;
        let end = slf.end;
        let fm_index = &slf.fm_index;
        let char_indices = &slf.char_indices;
        let (step, result) = py.detach(|| {
            let mut step = 0usize;
            while k + step < end {
                let byte_indice = fm_index.suffix_idx(k + step)?;
                if let Some(&indice) = char_indices.get(&byte_indice) {
                    return PyResult::Ok((step + 1, Some(indice)));
                }
                step += 1;
            }
            Ok((step, None))
        })?;

        slf.k += step;
        Ok(result)
    }
}

/// An FM-index for efficient full-text search on a single string.
///
/// The FM-index is a compressed text index based on the Burrows–Wheeler Transform (BWT).  
/// It supports fast substring queries whose runtime depends only on the pattern length,  
/// not on the size of the indexed text.
///
/// ### Construction
/// #### Time / Space Complexity
/// - Time: `O(N log σ)`
/// - Space: `O(N log σ)`
///
/// where:
/// - `N` = length of the indexed string
/// - `σ` = size of the alphabet (2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.)
///
/// ```python
/// from fm_index import FMIndex
///
/// fm = FMIndex("mississippi")
/// ```
#[derive(Clone)]
#[pyclass(name = "FMIndex")]
pub(crate) struct PyFMIndex {
    str_len: usize,
    fm_index: sync::Arc<FMIndex>,
    char_indices: sync::Arc<collections::HashMap<usize, usize>>,
}

#[pymethods]
impl PyFMIndex {
    /// Create a FM-Index from the given string.
    #[new]
    fn new(py: Python<'_>, data: &Bound<'_, PyString>) -> PyResult<Self> {
        let str = data.to_str()?;
        py.detach(|| {
            let str_len = str.chars().count();
            let bytes = str.as_bytes();
            let fm_index = FMIndex::new(bytes)?;
            let char_indices = str
                .char_indices()
                .enumerate()
                .map(|(char_idx, (byte_idx, _))| (byte_idx, char_idx))
                .chain(iter::once((bytes.len(), str_len)))
                .collect::<collections::HashMap<usize, usize>>();

            Ok(Self {
                str_len,
                fm_index: sync::Arc::new(fm_index),
                char_indices: sync::Arc::new(char_indices),
            })
        })
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.str_len)
    }

    fn __contains__(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        self.contains(py, pattern)
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let str = py.detach(|| {
            let bytes = self.fm_index.values()?;
            PyResult::Ok(format!(
                "FMIndex(\"{:}\")",
                String::from_utf8(bytes).map_err(PyUnicodeDecodeError::new_err)?
            ))
        })?;
        Ok(PyString::new(py, &str).into())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        self.__str__(py)
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| Ok(self.clone()))
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__copy__(py)
    }

    /// Convert the FMIndex back into the original string.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(N log σ)`
    /// - Space: `O(N)`
    ///
    /// #### Examples
    /// ```python
    /// fm.item()
    /// # 'mississippi'
    /// ```
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let bytes = py.detach(|| self.fm_index.values())?;
        Ok(PyString::from_bytes(py, &bytes)?.into())
    }

    /// Check whether the indexed string contains the given pattern.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern| log σ)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.contains("issi")
    /// # True
    /// ```
    fn contains(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        let pattern = pattern.to_str()?;
        py.detach(|| {
            let pattern = pattern.as_bytes();
            self.fm_index.contains(pattern)
        })
    }

    /// Count how many times a pattern appears in the indexed string.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern| log σ)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.count("issi")
    /// # 2
    /// ```
    fn count(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<usize> {
        let pattern = pattern.to_str()?;
        if pattern.is_empty() {
            // Special case: empty pattern
            return Ok(self.str_len + 1);
        }
        py.detach(|| {
            let pattern = pattern.as_bytes();
            self.fm_index.count(pattern)
        })
    }

    /// Locate all starting positions of the pattern in the indexed string.  
    /// ⚠️ Order of returned positions is not guaranteed.
    ///
    /// #### Complexity
    ///
    /// - Time: `O((|pattern| + |count|) log σ)`
    /// - Space: `O(|pattern| + |count|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.locate("issi")
    /// # [4, 1]
    /// ```
    fn locate(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<Py<PyList>> {
        let pattern = pattern.to_str()?;
        let char_locate = py.detach(|| {
            let pattern = pattern.as_bytes();
            let byte_locate = self.fm_index.locate(pattern)?;
            let char_locate = byte_locate
                .iter()
                .filter_map(|byte_idx| self.char_indices.get(byte_idx).copied())
                .collect::<Vec<_>>();
            PyResult::Ok(char_locate)
        })?;
        Ok(PyList::new(py, &char_locate)?.unbind())
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
    /// - Time: `O(|pattern| log σ)` for initialization, `O(log σ)` per yielded position
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
        let (start, end) = py.detach(|| self.fm_index.range_search(pattern.as_bytes()))?;
        Ok(IterLocate {
            k: start,
            end,
            fm_index: self.fm_index.clone(),
            char_indices: self.char_indices.clone(),
        })
    }

    /// Check if the indexed string starts with the given prefix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|prefix| log σ)`
    /// - Space: `O(|prefix|)`
    ///
    /// #### Examples
    /// ```python
    /// fm.startswith("mi")
    /// # True
    /// ```
    fn startswith(&self, py: Python<'_>, prefix: &Bound<'_, PyString>) -> PyResult<bool> {
        let prefix = prefix.to_str()?;
        py.detach(|| {
            let prefix = prefix.as_bytes();
            self.fm_index.starts_with(prefix)
        })
    }

    /// Check if the indexed string ends with the given suffix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|suffix| log σ)`
    /// - Space: `O(|suffix|)`
    ///
    /// where:
    /// - `|suffix|` = length of the suffix
    /// - `σ` = size of the alphabet (2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.)
    ///
    /// #### Examples
    /// ```python
    /// fm.endswith("pi")
    /// # True
    /// ```
    fn endswith(&self, py: Python<'_>, suffix: &Bound<'_, PyString>) -> PyResult<bool> {
        let suffix = suffix.to_str()?;
        py.detach(|| {
            let suffix = suffix.as_bytes();
            self.fm_index.ends_with(suffix)
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
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "")).unwrap();

            assert_eq!(fm_index.__len__().unwrap(), 0);
            assert_eq!(
                fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "FMIndex(\"\")"
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
            let iter_locate = fm_index.iter_locate(py, &PyString::new(py, "")).unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            assert_eq!(
                IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap(),
                Some(0)
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
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "mississippi")).unwrap();

            assert_eq!(fm_index.__len__().unwrap(), 11);
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
                fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "FMIndex(\"mississippi\")"
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
                PyFMIndex::new(py, &PyString::new(py, "にわにはにわにわとりがいる")).unwrap();

            assert_eq!(fm_index.__len__().unwrap(), 13);
            assert_eq!(
                fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "FMIndex(\"にわにはにわにわとりがいる\")"
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
                PyFMIndex::new(py, &PyString::new(py, "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊")).unwrap();

            assert_eq!(fm_index.__len__().unwrap(), 15);
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
                fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "FMIndex(\"🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊\")"
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
            ); // "⚔️" counts as 2 letters
            assert_eq!(
                fm_index
                    .locate(py, &PyString::new(py, "にわ"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
            ); // "⚔️" counts as 2 letters
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
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦👨‍👧")).unwrap();

            assert_eq!(fm_index.__len__().unwrap(), 35);
            assert!(fm_index.__contains__(py, &PyString::new(py, "👨‍👩‍👧‍👦")).unwrap());
            assert_eq!(
                fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "FMIndex(\"👨‍👩‍👧‍👦👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦👨‍👧\")"
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
