use std::char;

use pyo3::{
    PyResult,
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyList, PyString, PyStringData, PyStringMethods},
};

use crate::fm_index::fm_index::FMIndex;

#[derive(Clone)]
enum FMIndexEnum {
    U8(FMIndex<u8>),
    U16(FMIndex<u16>),
    U32(FMIndex<u32>),
}

/// An FM-index data structure for efficient substring search on text.
///
/// The FM-index is a compressed full-text index built on the
/// Burrows–Wheeler Transform (BWT) and succinct rank/select structures.
/// It supports fast pattern counting and locating while keeping memory usage low.
///
/// This implementation accepts Unicode strings and internally encodes the input
/// as a sequence of integer symbols. Depending on the input string’s representation
/// (e.g., UCS-1 / UCS-2 / UCS-4), it automatically selects an appropriate integer
/// bit-width for symbols, using the smallest representation that can faithfully
/// store all code points in the text.
#[derive(Clone)]
#[pyclass(name = "FMIndex")]
pub(crate) struct PyFMIndex {
    inner: FMIndexEnum,
}

#[pymethods]
impl PyFMIndex {
    /// Create a FM-Index from the given string.
    #[new]
    fn new(py: Python<'_>, data: &Bound<'_, PyString>) -> PyResult<Self> {
        let data = unsafe { data.data()? };
        py.detach(move || {
            let fm_index = match data {
                PyStringData::Ucs1(data) => FMIndexEnum::U8(FMIndex::new(data)?),
                PyStringData::Ucs2(data) => FMIndexEnum::U16(FMIndex::new(data)?),
                PyStringData::Ucs4(data) => FMIndexEnum::U32(FMIndex::new(data)?),
            };
            Ok(PyFMIndex { inner: fm_index })
        })
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(move || match &self.inner {
            FMIndexEnum::U8(fm_index) => fm_index.len(),
            FMIndexEnum::U16(fm_index) => fm_index.len(),
            FMIndexEnum::U32(fm_index) => fm_index.len(),
        })
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(move || match &self.inner {
            FMIndexEnum::U8(fm_index) => Ok(format!(
                "FMIndex(\"{:}\")",
                String::from_utf8(fm_index.values()?).map_err(PyRuntimeError::new_err)?
            )),
            FMIndexEnum::U16(fm_index) => Ok(format!(
                "FMIndex(\"{:}\")",
                String::from_utf16(&fm_index.values()?).map_err(PyRuntimeError::new_err)?
            )),
            FMIndexEnum::U32(fm_index) => Ok(format!(
                "FMIndex(\"{:}\")",
                fm_index
                    .values()?
                    .iter()
                    .map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
                    .collect::<String>()
            )),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(move || match &self.inner {
            FMIndexEnum::U8(fm_index) => Ok(format!(
                "FMIndex(\"{:}\")",
                String::from_utf8(fm_index.values()?).map_err(PyRuntimeError::new_err)?
            )),
            FMIndexEnum::U16(fm_index) => Ok(format!(
                "FMIndex(\"{:}\")",
                String::from_utf16(&fm_index.values()?).map_err(PyRuntimeError::new_err)?
            )),
            FMIndexEnum::U32(fm_index) => Ok(format!(
                "FMIndex(\"{:}\")",
                fm_index
                    .values()?
                    .iter()
                    .map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
                    .collect::<String>()
            )),
        })
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(move || Ok(self.clone()))
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        py.detach(move || Ok(self.clone()))
    }

    /// Convert the FM-Index back to a string.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(N log σ)`
    ///
    /// where:
    /// - `N` = length of the indexed string
    /// - `σ` = size of the alphabet (2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.)
    ///
    /// #### Examples
    /// ```python
    /// >>> from fm_index import FMIndex
    /// >>> fm_index = FMIndex("mississippi")
    /// >>> fm_index.item()
    /// "mississippi"
    /// ```
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyString>> {
        let str = py.detach(move || match &self.inner {
            FMIndexEnum::U8(fm_index) => {
                String::from_utf8(fm_index.values()?).map_err(PyRuntimeError::new_err)
            }
            FMIndexEnum::U16(fm_index) => {
                String::from_utf16(&fm_index.values()?).map_err(PyRuntimeError::new_err)
            }
            FMIndexEnum::U32(fm_index) => Ok(fm_index
                .values()?
                .iter()
                .map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
                .collect::<String>()),
        })?;
        Ok(PyString::new(py, &str).into())
    }

    /// Count the occurrences of the given pattern in the indexed string.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|pattern| log σ)`
    ///
    /// where:
    /// - `|pattern|` = length of the pattern
    /// - `σ` = size of the alphabet (2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.)
    ///
    /// #### Examples
    /// ```python
    /// >>> from fm_index import FMIndex
    /// >>> fm_index = FMIndex("mississippi")
    /// >>> fm_index.count("issi")
    /// 2
    /// ```
    fn count(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<usize> {
        let pattern = unsafe { pattern.data()? };
        py.detach(move || match &self.inner {
            FMIndexEnum::U8(fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => data,
                    _ => return Ok(0usize),
                };
                fm_index.count(pattern)
            }
            FMIndexEnum::U16(fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => &data.iter().map(|&c| c as u16).collect::<Vec<_>>(),
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(0usize),
                };
                fm_index.count(pattern)
            }
            FMIndexEnum::U32(fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => &data.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                    PyStringData::Ucs2(data) => &data.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                    PyStringData::Ucs4(data) => data,
                };
                fm_index.count(pattern)
            }
        })
    }

    /// Locate all occurrences of the given pattern in the indexed string.
    /// Order of the returned positions is not guaranteed.
    ///
    /// #### Complexity
    ///
    /// - Time: `O((|pattern| + |count|) log σ)`
    ///
    /// where:
    /// - `|pattern|` = length of the pattern
    /// - `|count|` = number of occurrences of the pattern in the indexed string
    /// - `σ` = size of the alphabet (2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.)
    ///
    /// #### Examples
    /// ```python
    /// >>> from fm_index import FMIndex
    /// >>> fm_index = FMIndex("mississippi")
    /// >>> fm_index.locate("issi")
    /// [4, 1]
    /// ```
    fn locate(&self, py: Python<'_>, pattern: &Bound<'_, PyString>) -> PyResult<Py<PyList>> {
        let pattern = unsafe { pattern.data()? };
        let locate = py.detach(move || match &self.inner {
            FMIndexEnum::U8(fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => data,
                    _ => return PyResult::Ok(vec![]),
                };
                Ok(fm_index.locate(pattern)?)
            }
            FMIndexEnum::U16(fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => &data.iter().map(|&c| c as u16).collect::<Vec<_>>(),
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(vec![]),
                };
                Ok(fm_index.locate(pattern)?)
            }
            FMIndexEnum::U32(fm_index) => {
                let pattern = match pattern {
                    PyStringData::Ucs1(data) => &data.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                    PyStringData::Ucs2(data) => &data.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                    PyStringData::Ucs4(data) => data,
                };
                Ok(fm_index.locate(pattern)?)
            }
        })?;
        Ok(PyList::new(py, &locate)?.unbind())
    }

    /// Check if the indexed string starts with the given prefix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|prefix| log σ)`
    ///
    /// where:
    /// - `|prefix|` = length of the prefix
    /// - `σ` = size of the alphabet (2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.)
    ///
    /// #### Examples
    /// ```python
    /// >>> from fm_index import FMIndex
    /// >>> fm_index = FMIndex("mississippi")
    /// >>> fm_index.starts_with("mi")
    /// True
    /// ```
    fn startswith(&self, py: Python<'_>, prefix: &Bound<'_, PyString>) -> PyResult<bool> {
        let prefix = unsafe { prefix.data()? };
        py.detach(move || match &self.inner {
            FMIndexEnum::U8(fm_index) => {
                let prefix = match prefix {
                    PyStringData::Ucs1(data) => data,
                    _ => return Ok(false),
                };
                fm_index.starts_with(prefix)
            }
            FMIndexEnum::U16(fm_index) => {
                let prefix = match prefix {
                    PyStringData::Ucs1(data) => &data.iter().map(|&c| c as u16).collect::<Vec<_>>(),
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(false),
                };
                fm_index.starts_with(prefix)
            }
            FMIndexEnum::U32(fm_index) => {
                let prefix = match prefix {
                    PyStringData::Ucs1(data) => &data.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                    PyStringData::Ucs2(data) => &data.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                    PyStringData::Ucs4(data) => data,
                };
                fm_index.starts_with(prefix)
            }
        })
    }

    /// Check if the indexed string ends with the given suffix.
    ///
    /// #### Complexity
    ///
    /// - Time: `O(|suffix| log σ)`
    ///
    /// where:
    /// - `|suffix|` = length of the suffix
    /// - `σ` = size of the alphabet (2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.)
    ///
    /// #### Examples
    /// ```python
    /// >>> from fm_index import FMIndex
    /// >>> fm_index = FMIndex("mississippi")
    /// >>> fm_index.ends_with("pi")
    /// True
    /// ```
    fn endswith(&self, py: Python<'_>, suffix: &Bound<'_, PyString>) -> PyResult<bool> {
        let suffix = unsafe { suffix.data()? };
        py.detach(move || match &self.inner {
            FMIndexEnum::U8(fm_index) => {
                let suffix = match suffix {
                    PyStringData::Ucs1(data) => data,
                    _ => return Ok(false),
                };
                fm_index.ends_with(suffix)
            }
            FMIndexEnum::U16(fm_index) => {
                let suffix = match suffix {
                    PyStringData::Ucs1(data) => &data.iter().map(|&c| c as u16).collect::<Vec<_>>(),
                    PyStringData::Ucs2(data) => data,
                    _ => return Ok(false),
                };
                fm_index.ends_with(suffix)
            }
            FMIndexEnum::U32(fm_index) => {
                let suffix = match suffix {
                    PyStringData::Ucs1(data) => &data.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                    PyStringData::Ucs2(data) => &data.iter().map(|&c| c as u32).collect::<Vec<_>>(),
                    PyStringData::Ucs4(data) => data,
                };
                fm_index.ends_with(suffix)
            }
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

            assert_eq!(fm_index.__str__(py).unwrap(), "FMIndex(\"\")");
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                ""
            );
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

            assert_eq!(fm_index.__str__(py).unwrap(), "FMIndex(\"mississippi\")");
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "mississippi"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 12);
            assert_eq!(fm_index.count(py, &PyString::new(py, "issi")).unwrap(), 2);
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
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.startswith(py, &PyString::new(py, "miss")).unwrap());
            assert!(
                !fm_index
                    .startswith(py, &PyString::new(py, "いっぴ"))
                    .unwrap()
            );
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "ippi")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "ミス")).unwrap());
        });
    }

    #[test]
    fn test_fm_index_ucs2() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index =
                PyFMIndex::new(py, &PyString::new(py, "にわにはにわにわとりがいる")).unwrap();

            assert_eq!(
                fm_index.__str__(py).unwrap(),
                "FMIndex(\"にわにはにわにわとりがいる\")"
            );
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "にわにはにわにわとりがいる"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 14);
            assert_eq!(fm_index.count(py, &PyString::new(py, "にわ")).unwrap(), 3);
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
                    .locate(py, &PyString::new(py, "にわ"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [6, 0, 4]
            );
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
            assert!(
                fm_index
                    .startswith(py, &PyString::new(py, "にわに"))
                    .unwrap()
            );
            assert!(!fm_index.startswith(py, &PyString::new(py, "🐓")).unwrap());
            assert!(fm_index.endswith(py, &PyString::new(py, "")).unwrap());
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

            assert_eq!(
                fm_index.__str__(py).unwrap(),
                "FMIndex(\"🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊\")"
            );
            assert_eq!(
                fm_index.item(py).unwrap().extract::<String>(py).unwrap(),
                "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊"
            );
            assert_eq!(fm_index.count(py, &PyString::new(py, "")).unwrap(), 16);
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
                    .locate(py, &PyString::new(py, "🐉🔥🌊"))
                    .unwrap()
                    .extract::<Vec<usize>>(py)
                    .unwrap(),
                [12, 6, 1]
            ); // "⚔️" counts as 2 letters
            assert!(fm_index.startswith(py, &PyString::new(py, "")).unwrap());
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
            assert!(fm_index.endswith(py, &PyString::new(py, "🐉🔥🌊")).unwrap());
            assert!(!fm_index.endswith(py, &PyString::new(py, "⚔️🐉🔥")).unwrap());
        });
    }

    #[test]
    fn test_fm_index_zwj() {
        Python::initialize();

        Python::attach(|py| {
            let fm_index = PyFMIndex::new(py, &PyString::new(py, "👨‍👩‍👧‍👦👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦👨‍👧")).unwrap();

            assert_eq!(fm_index.__str__(py).unwrap(), "FMIndex(\"👨‍👩‍👧‍👦👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦xx👨‍👩‍👧‍👦👨‍👧\")");
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
