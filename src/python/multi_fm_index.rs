use std::sync;

use pyo3::{
    PyResult,
    exceptions::PyValueError,
    prelude::*,
    types::{IntoPyDict, PyBytes, PyBytesMethods, PyList, PySequence, PyString, PyStringMethods},
};

use crate::fm_index::multi_fm_index::{iter_locate::IterLocate, multi_fm_index::MultiFMIndex};

/// A multi-document FM-index for fast substring search across multiple strings.
///
/// Internally, all strings are concatenated with separators and indexed as a single FM-index,
/// while preserving the ability to map matches back to their original documents.
/// Query processing across documents is internally parallelized where applicable,
/// making multi-document search efficient in practice.
///
/// ### Construction
/// #### Time / Space Complexity
/// - Time: `O(S + D log D)`
/// - Space: `O(S + D log D)`
///
/// where:
/// - `D` = number of indexed strings
/// - `S` = total length of all indexed strings
///
/// ```python
/// from fm_index import MultiFMIndex
///
/// mfm = MultiFMIndex(["abcabcabcabc", "xxabcabcxxabc", "abcababcabc"])
/// ```
///
/// ### Serialization
/// MultiFMIndex supports Python's pickle protocol for efficient persistence:
///
/// ```python
/// import pickle
///
/// # Save index
/// with open("index.pkl", "wb") as f:
///     pickle.dump(mfm, f)
///
/// # Load index
/// with open("index.pkl", "rb") as f:
///     mfm = pickle.load(f)
/// ```
#[derive(Clone)]
#[pyclass(name = "MultiFMIndex", module = "fm_index")]
pub(crate) struct PyMultiFMIndex {
    inner: sync::Arc<MultiFMIndex>,
}

#[pymethods]
impl PyMultiFMIndex {
    /// Create a MultiFMIndex from the given list of strings.
    #[new]
    fn new(py: Python<'_>, data: &Bound<'_, PySequence>) -> PyResult<Self> {
        let data = data
            .try_iter()?
            .map(|item| {
                let bound_item = item?;
                let item = bound_item.cast::<PyString>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "All elements in the sequence must be strings.",
                    )
                })?;
                Ok(item.to_string())
            })
            .collect::<PyResult<Vec<_>>>()?;
        py.detach(move || {
            let inner = MultiFMIndex::new(data)?;
            Ok(PyMultiFMIndex {
                inner: sync::Arc::new(inner),
            })
        })
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.inner.num_docs())
    }

    fn __contains__(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        self.contains(py, pattern)
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let (num_docs, total_num_chars, max_bit) =
            py.detach(|| -> PyResult<(usize, usize, usize)> {
                let num_docs = self.inner.num_docs()?;
                let total_num_chars = self.inner.total_num_chars()?;
                let max_bit = self.inner.max_bit()?;
                Ok((num_docs, total_num_chars, max_bit))
            })?;
        let result = format!(
            "MultiFMIndex(num_docs={}, total_num_chars={}, max_bit={})",
            num_docs, total_num_chars, max_bit,
        );
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
            Ok(PyMultiFMIndex {
                inner: sync::Arc::new((*self.inner).clone()),
            })
        })
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        // Return (class, args, state) where:
        // - class: the class to instantiate
        // - args: arguments for __new__ (empty list for us)
        // - state: will be passed to __setstate__
        let cls = py.import("fm_index")?.getattr("MultiFMIndex")?.into();
        let args = (PyList::empty(py),).into_pyobject(py)?.into_any().unbind();
        let state: Py<PyAny> = self.__getstate__(py)?.into();
        Ok((cls, args, state))
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let serialized = postcard::to_allocvec(&*self.inner)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize: {}", e)))?;
        Ok(PyBytes::new(py, &serialized).into())
    }

    fn __setstate__(&mut self, _py: Python<'_>, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = state.as_bytes();
        let inner: MultiFMIndex = postcard::from_bytes(bytes)
            .map_err(|e| PyValueError::new_err(format!("Failed to deserialize: {}", e)))?;
        self.inner = sync::Arc::new(inner);
        Ok(())
    }

    /// Convert the index back into the original list of strings.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(S)`
    /// - Space: `O(S)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.item()
    /// # ['abcabcabcabc', 'xxabcabcxxabc', 'abcababcabc']
    /// ```
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let str_list = py.detach(|| self.inner.values())?;
        Ok(PyList::new(py, str_list)?.unbind())
    }

    /// Check if the pattern exists as a full document in the index.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern|)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.contains("abcabcabcabc")
    /// # True
    /// ```
    fn contains(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<bool> {
        let pattern = pattern.to_str()?;
        py.detach(|| self.inner.contains(pattern))
    }

    /// Count total occurrences of a pattern across all documents.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern|)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.count_all("abc")
    /// # 10
    /// ```
    fn count_all(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<usize> {
        let pattern = pattern.to_str()?;
        py.detach(|| self.inner.count_all(pattern))
    }

    /// Count occurrences per document or within a specific document.
    ///
    /// #### Complexity
    ///
    /// When `doc_id` is None:
    /// - Time: `O(|pattern| + |output| log D)`
    /// - Space: `O(|pattern| + |output|)`
    ///
    /// When `doc_id` is specified:
    /// - Time: `O(|pattern| + log D)`
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// # Count across all documents (returns {doc_index: count})
    /// mfm.count("abc")
    /// # {0: 4, 1: 3, 2: 3}
    ///
    /// # Count within a specific document (returns int)
    /// mfm.count("abc", doc_id=0)
    /// # 4
    /// ```
    #[pyo3(signature = (pattern, doc_id=None))]
    fn count(
        &self,
        py: Python<'_>,
        pattern: &Bound<'_, PyString>,
        doc_id: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let pattern = pattern.to_str()?;
        match doc_id {
            Some(doc_id) => {
                let count = py.detach(|| self.inner.count_within_doc(doc_id, pattern))?;
                Ok(count.into_pyobject(py)?.into_any().unbind())
            }
            None => {
                let count = py.detach(|| self.inner.count(pattern))?;
                Ok(count.into_py_dict(py)?.into_any().unbind())
            }
        }
    }

    /// Return the top-k documents with the highest number of occurrences
    /// of the given pattern.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern| + |total_count| log |total_count| log D)`
    /// - Space: `O(|pattern| + k)`
    ///
    /// #### Examples
    /// ```python
    /// # Get top 2 documents containing the pattern
    /// mfm.topk("abc", k=2)
    /// # [(0, 4), (1, 3)]
    ///
    /// # If fewer than k documents match, all matching documents are returned
    /// mfm.topk("abc", k=5)
    /// # [(0, 4), (1, 3), (2, 3)]
    /// ```
    fn topk(
        &self,
        py: Python<'_>,
        pattern: &Bound<'_, PyString>,
        k: usize,
    ) -> PyResult<Py<PyList>> {
        let pattern = pattern.to_str()?;
        let result = py.detach(|| self.inner.topk(pattern, k))?;
        Ok(PyList::new(py, result)?.unbind())
    }

    /// Locate occurrences per document or within a specific document.
    /// Internally, result enumeration and aggregation may be parallelized.
    /// ⚠️ Order is not guaranteed.
    ///
    /// #### Complexity
    ///
    /// When `doc_id` is None:
    /// - Time: `O(|pattern| + |total_count| log D)`
    /// - Space: `O(|pattern| + |total_count|)`
    ///
    /// When `doc_id` is specified:
    /// - Time: `O(|pattern| + |count_in_doc| log D)`
    /// - Space: `O(|pattern| + |count_in_doc|)`
    ///
    /// #### Examples
    /// ```python
    /// # Locate across all documents (returns {doc_index: [positions]})
    /// mfm.locate("abc")
    /// # {0: [9, 6, 3, 0], 1: [10, 2, 5], 2: [8, 0, 5]}
    ///
    /// # Locate within a specific document (returns list[int])
    /// mfm.locate("abc", doc_id=0)
    /// # [9, 6, 3, 0]
    /// ```
    #[pyo3(signature = (pattern, doc_id=None))]
    fn locate(
        &self,
        py: Python<'_>,
        pattern: &Bound<'_, PyString>,
        doc_id: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let pattern = pattern.to_str()?;
        match doc_id {
            Some(doc_id) => {
                let locate = py.detach(|| self.inner.locate_within_doc(doc_id, pattern))?;
                Ok(locate.into_pyobject(py)?.into_any().unbind())
            }
            None => {
                let locate = py.detach(|| self.inner.locate(pattern))?;
                Ok(locate.into_py_dict(py)?.into_any().unbind())
            }
        }
    }

    /// Lazily locate all occurrences of the pattern across documents or within a specific document.
    ///
    /// When `doc_id` is None, yields `(doc_id, position)` pairs.
    /// When `doc_id` is specified, yields only `position` values.
    ///
    /// ⚠️ Order of yielded results is not guaranteed.
    ///
    /// ### Complexity
    ///
    /// When `doc_id` is None:
    /// - Time: `O(|pattern|)` to initialize, then `O(log D)` per yielded occurrence.
    /// - Space: `O(|pattern|)`
    ///
    /// When `doc_id` is specified:
    /// - Time: `O(|pattern| + log D)` to initialize, then `O(log D)` per yielded occurrence within the document.
    /// - Space: `O(|pattern|)`
    ///
    /// #### Examples
    /// ```python
    /// # Iterate across all documents
    /// iter = mfm.iter_locate("abc")
    /// next(iter)
    /// # (2, 8)
    /// next(iter)
    /// # (1, 10)
    ///
    /// # Iterate within a specific document
    /// iter = mfm.iter_locate("abc", doc_id=0)
    /// next(iter)
    /// # 9
    /// next(iter)
    /// # 6
    /// ...
    /// ```
    #[pyo3(signature = (pattern, doc_id=None))]
    fn iter_locate(
        &self,
        py: Python<'_>,
        pattern: &Bound<'_, PyString>,
        doc_id: Option<usize>,
    ) -> PyResult<IterLocate> {
        let pattern = pattern.to_str()?;
        let inner = self.inner.clone();
        py.detach(move || IterLocate::new(doc_id, pattern, inner))
    }

    /// List document indices whose content starts with the prefix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|prefix| + |output|)`
    /// - Space: `O(|prefix| + |output|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.startswith("abc")
    /// # [2, 0]
    /// ```
    fn startswith(&self, py: Python<'_>, prefix: &Bound<'_, PyString>) -> PyResult<Py<PyList>> {
        let prefix = prefix.to_str()?;
        let result = py.detach(|| self.inner.starts_with(prefix))?;
        Ok(PyList::new(py, result)?.unbind())
    }

    /// List document indices whose content ends with the suffix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|suffix| + |output| log D)`
    /// - Space: `O(|suffix| + |output|)`
    ///
    /// #### Examples
    /// ```python
    /// mfm.endswith("abc")
    /// # [2, 1, 0]
    /// ```
    fn endswith(&self, py: Python<'_>, suffix: &Bound<'_, PyString>) -> PyResult<Py<PyList>> {
        let suffix = suffix.to_str()?;
        let result = py.detach(|| self.inner.ends_with(suffix))?;
        Ok(PyList::new(py, result)?.unbind())
    }
}

#[cfg(test)]
mod tests {
    use std::collections;

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
                "MultiFMIndex(num_docs=0, total_num_chars=0, max_bit=0)",
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
                    .count(py, &PyString::new(py, ""), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "a"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, ""), Some(0))
                    .unwrap_err()
                    .to_string(),
                "ValueError: doc_id is out of bounds",
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "a"), Some(0))
                    .unwrap_err()
                    .to_string(),
                "ValueError: doc_id is out of bounds",
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "a"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""), Some(0))
                    .unwrap_err()
                    .to_string(),
                "ValueError: doc_id is out of bounds",
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "a"), Some(0))
                    .unwrap_err()
                    .to_string(),
                "ValueError: doc_id is out of bounds",
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
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "a"), 1)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                Vec::<(usize, usize)>::new()
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "a"), 0)
                    .unwrap_err()
                    .to_string(),
                "ValueError: k must be greater than 0"
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
                "MultiFMIndex(num_docs=3, total_num_chars=0, max_bit=0)",
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
                    .count(py, &PyString::new(py, ""), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::from([(0, 1), (1, 1), (2, 1)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "a"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, ""), Some(0))
                    .unwrap()
                    .extract::<usize>(py)
                    .unwrap(),
                1,
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "a"), Some(0))
                    .unwrap()
                    .extract::<usize>(py)
                    .unwrap(),
                0,
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::from([(0, [0]), (1, [0]), (2, [0])])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "a"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""), Some(0))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                vec![0],
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "a"), Some(0))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                Vec::<usize>::new(),
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
            let mut result = multi_fm_index
                .topk(py, &PyString::new(py, ""), 2)
                .unwrap()
                .extract::<Vec<(usize, usize)>>(py)
                .unwrap();
            result.sort_by_key(|(doc_id, _)| *doc_id);
            assert_eq!(result, vec![(0, 1), (1, 1)]);
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "a"), 1)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                Vec::<(usize, usize)>::new()
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
                "MultiFMIndex(num_docs=3, total_num_chars=36, max_bit=7)",
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
                    .count(py, &PyString::new(py, ""), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 13), (1, 14), (2, 12)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "abc"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 4), (1, 3), (2, 3)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "あいう"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "😀😃😀"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""), None)
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
                    .locate(py, &PyString::new(py, "abc"), None)
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
                    .locate(py, &PyString::new(py, "あいう"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "😀😃😀"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            let iter_locate = multi_fm_index
                .iter_locate(py, &PyString::new(py, "abc"), None)
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            let mut results = vec![];
            while let Some(result) = IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap() {
                results.push(result);
            }
            assert_eq!(results.len(), 10);
            assert_eq!(results[0].extract::<(usize, usize)>(py).unwrap(), (2, 8));
            let iter_locate_doc = multi_fm_index
                .iter_locate(py, &PyString::new(py, "abc"), Some(0))
                .unwrap();
            let py_iter_doc = Py::new(py, iter_locate_doc).unwrap();
            let mut results = vec![];
            while let Some(result) = IterLocate::__next__(py_iter_doc.borrow_mut(py), py).unwrap() {
                results.push(result);
            }
            let extracted: Vec<usize> = results
                .into_iter()
                .map(|item| item.extract::<usize>(py).unwrap())
                .collect();
            assert_eq!(extracted, vec![9, 6, 3, 0]);
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
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "abc"), 2)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 4), (1, 3)]
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "abc"), 5)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 4), (1, 3), (2, 3)]
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "abc"), 1)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 4)]
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "xyz"), 2)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                Vec::<(usize, usize)>::new()
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
                "MultiFMIndex(num_docs=3, total_num_chars=27, max_bit=14)",
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
                    .count(py, &PyString::new(py, ""), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 10), (1, 11), (2, 9)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "abc"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "あいう"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 3), (1, 2), (2, 2)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "😀😃😀"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""), None)
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
                    .locate(py, &PyString::new(py, "abc"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "あいう"), None)
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
                    .locate(py, &PyString::new(py, "😀😃😀"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            let iter_locate = multi_fm_index
                .iter_locate(py, &PyString::new(py, "あいう"), None)
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            let mut results = vec![];
            while let Some(result) = IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap() {
                results.push(result);
            }
            assert_eq!(results.len(), 7);
            assert_eq!(results[0].extract::<(usize, usize)>(py).unwrap(), (2, 5));
            let iter_locate_doc = multi_fm_index
                .iter_locate(py, &PyString::new(py, "あいう"), Some(1))
                .unwrap();
            let py_iter_doc = Py::new(py, iter_locate_doc).unwrap();
            let mut results = vec![];
            while let Some(result) = IterLocate::__next__(py_iter_doc.borrow_mut(py), py).unwrap() {
                results.push(result);
            }
            let extracted: Vec<usize> = results
                .into_iter()
                .map(|item| item.extract::<usize>(py).unwrap())
                .collect();
            assert_eq!(extracted, vec![5, 2]);
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
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "あいう"), 2)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 3), (1, 2)]
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "あいう"), 5)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 3), (1, 2), (2, 2)]
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "abc"), 1)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                Vec::<(usize, usize)>::new()
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
                "MultiFMIndex(num_docs=3, total_num_chars=19, max_bit=17)",
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
                    .count(py, &PyString::new(py, ""), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 7), (1, 9), (2, 6)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "abc"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "あいう"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::new()
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "😀😃😀"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 2), (1, 1), (2, 1)])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""), None)
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
                    .locate(py, &PyString::new(py, "abc"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "あいう"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::new()
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, "😀😃😀"), None)
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
                .iter_locate(py, &PyString::new(py, "😀😃😀"), None)
                .unwrap();
            let py_iter = Py::new(py, iter_locate).unwrap();
            let mut results = vec![];
            while let Some(result) = IterLocate::__next__(py_iter.borrow_mut(py), py).unwrap() {
                results.push(result);
            }
            assert_eq!(results.len(), 4);
            assert_eq!(results[0].extract::<(usize, usize)>(py).unwrap(), (2, 0));
            let iter_locate_doc = multi_fm_index
                .iter_locate(py, &PyString::new(py, "😀😃😀"), Some(0))
                .unwrap();
            let py_iter_doc = Py::new(py, iter_locate_doc).unwrap();
            let mut results = vec![];
            while let Some(result) = IterLocate::__next__(py_iter_doc.borrow_mut(py), py).unwrap() {
                results.push(result);
            }
            let extracted: Vec<usize> = results
                .into_iter()
                .map(|item| item.extract::<usize>(py).unwrap())
                .collect();
            assert_eq!(extracted, vec![2, 0]);
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
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "😀😃😀"), 2)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 2), (1, 1)]
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "😀😃😀"), 5)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 2), (1, 1), (2, 1)]
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "abc"), 1)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                Vec::<(usize, usize)>::new()
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
                "MultiFMIndex(num_docs=3, total_num_chars=35, max_bit=17)",
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
                    .count(py, &PyString::new(py, ""), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 15), (1, 12), (2, 11)])
            );
            assert_eq!(
                multi_fm_index
                    .count(py, &PyString::new(py, "👨‍👩‍👧‍👦"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, usize>::from([(0, 2), (1, 1), (2, 1)])
            );
            assert_eq!(
                multi_fm_index
                    .locate(py, &PyString::new(py, ""), None)
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
                    .locate(py, &PyString::new(py, "👨‍👩‍👧‍👦"), None)
                    .unwrap()
                    .extract::<collections::HashMap<_, _>>(py)
                    .unwrap(),
                collections::HashMap::<usize, Vec<usize>>::from([
                    (0, vec![7, 0]),
                    (1, vec![2]),
                    (2, vec![0])
                ])
            );
            let iter_locate_doc = multi_fm_index
                .iter_locate(py, &PyString::new(py, "👨‍👩‍👧‍👦"), Some(0))
                .unwrap();
            let py_iter_doc = Py::new(py, iter_locate_doc).unwrap();
            let mut results = vec![];
            while let Some(result) = IterLocate::__next__(py_iter_doc.borrow_mut(py), py).unwrap() {
                results.push(result);
            }
            let extracted: Vec<usize> = results
                .into_iter()
                .map(|item| item.extract::<usize>(py).unwrap())
                .collect();
            assert_eq!(extracted, vec![7, 0]);
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
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "👨‍👩‍👧‍👦"), 2)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 2), (1, 1)]
            );
            assert_eq!(
                multi_fm_index
                    .topk(py, &PyString::new(py, "👨‍👩‍👧‍👦"), 5)
                    .unwrap()
                    .extract::<Vec<(usize, usize)>>(py)
                    .unwrap(),
                vec![(0, 2), (1, 1), (2, 1)]
            );
        });
    }
}
