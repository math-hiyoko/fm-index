use std::sync;

use pyo3::{
    PyResult,
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyBytesMethods, PyList, PyString, PyStringMethods},
};

use crate::fm_index::fm_index::{fm_index::FMIndex, iter_locate::IterLocate};

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
/// fm = FMIndex("mississippi")
/// ```
///
/// ### Serialization
/// FMIndex supports Python's pickle protocol for efficient persistence:
///
/// ```python
/// import pickle
///
/// # Save index
/// with open("index.pkl", "wb") as f:
///     pickle.dump(fm, f)
///
/// # Load index
/// with open("index.pkl", "rb") as f:
///     fm = pickle.load(f)
/// ```
#[derive(Clone)]
#[pyclass(name = "FMIndex", module = "fm_index")]
pub(crate) struct PyFMIndex {
    inner: sync::Arc<FMIndex>,
}

#[pymethods]
impl PyFMIndex {
    /// Create a FM-Index from the given string.
    #[new]
    fn new(py: Python<'_>, data: &Bound<'_, PyString>) -> PyResult<Self> {
        let data = data.to_str()?;
        py.detach(move || {
            let inner = FMIndex::new(data)?;
            Ok(PyFMIndex {
                inner: sync::Arc::new(inner),
            })
        })
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.inner.len())
    }

    fn __contains__(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        self.contains(py, pattern)
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let (len, max_bit) = py.detach(|| -> PyResult<(usize, usize)> {
            let len = self.inner.len()?;
            let max_bit = self.inner.max_bit()?;
            Ok((len, max_bit))
        })?;
        let result = format!("FMIndex(len={}, max_bit={})", len, max_bit,);
        Ok(PyString::new(py, &result).into())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        self.__str__(py)
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(move || Ok(self.clone()))
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        py.detach(move || {
            Ok(PyFMIndex {
                inner: sync::Arc::new((*self.inner).clone()),
            })
        })
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
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

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let serialized = postcard::to_allocvec(&*self.inner)
            .map_err(|error| PyValueError::new_err(format!("Failed to serialize: {}", error)))?;
        Ok(PyBytes::new(py, &serialized).into())
    }

    fn __setstate__(&mut self, _py: Python<'_>, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = state.as_bytes();
        let inner: FMIndex = postcard::from_bytes(bytes)
            .map_err(|error| PyValueError::new_err(format!("Failed to deserialize: {}", error)))?;
        self.inner = sync::Arc::new(inner);
        Ok(())
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
        let str = py.detach(|| self.inner.value())?;
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
        let pattern = pattern.to_str()?;
        py.detach(|| self.inner.contains(pattern))
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
        py.detach(|| self.inner.count(pattern))
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
        let locate = py.detach(|| self.inner.locate(pattern))?;
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
        let inner = self.inner.clone();
        py.detach(move || IterLocate::new(pattern, inner))
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
        py.detach(|| self.inner.starts_with(prefix))
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
        py.detach(|| self.inner.ends_with(suffix))
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

            assert_eq!(fm_index.__len__(py).unwrap(), 0);
            assert_eq!(
                fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "FMIndex(len=0, max_bit=0)"
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
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "mississippi")).unwrap();

            assert_eq!(fm_index.__len__(py).unwrap(), 11);
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
                "FMIndex(len=11, max_bit=7)"
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

            assert_eq!(fm_index.__len__(py).unwrap(), 13);
            assert_eq!(
                fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "FMIndex(len=13, max_bit=14)"
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

            assert_eq!(fm_index.__len__(py).unwrap(), 15);
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
                "FMIndex(len=15, max_bit=17)"
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

            assert_eq!(fm_index.__len__(py).unwrap(), 35);
            assert!(fm_index.__contains__(py, &PyString::new(py, "👨‍👩‍👧‍👦")).unwrap());
            assert_eq!(
                fm_index
                    .__repr__(py)
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap(),
                "FMIndex(len=35, max_bit=17)"
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
