use std::sync;

use pyo3::{PyResult, exceptions::PyUnicodeDecodeError, types::PyStringData};

use super::fm_index::FMIndex;

#[derive(Clone)]
pub(crate) enum FMIndexEnum {
    Ucs1(sync::Arc<FMIndex<u8>>),
    Ucs2(sync::Arc<FMIndex<u16>>),
    Ucs4(sync::Arc<FMIndex<u32>>),
}

impl FMIndexEnum {
    pub(crate) fn new(data: PyStringData) -> PyResult<Self> {
        match data {
            PyStringData::Ucs1(data) => Ok(FMIndexEnum::Ucs1(sync::Arc::new(FMIndex::new(
                data.to_vec(),
            )?))),
            PyStringData::Ucs2(data) => Ok(FMIndexEnum::Ucs2(sync::Arc::new(FMIndex::new(
                data.to_vec(),
            )?))),
            PyStringData::Ucs4(data) => Ok(FMIndexEnum::Ucs4(sync::Arc::new(FMIndex::new(
                data.to_vec(),
            )?))),
        }
    }

    pub(super) fn range_search(&self, pattern: PyStringData) -> PyResult<(usize, usize)> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => fm_index.range_search(pattern.to_vec()),
                _ => Ok((0, 0)),
            },
            FMIndexEnum::Ucs2(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.range_search(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => fm_index.range_search(pattern.to_vec()),
                _ => Ok((0, 0)),
            },
            FMIndexEnum::Ucs4(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.range_search(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    fm_index.range_search(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => fm_index.range_search(pattern.to_vec()),
            },
        }
    }

    pub(super) fn suffix_idx(&self, k: usize) -> PyResult<usize> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => {
                let value = fm_index.suffix_idx(k)?;
                Ok(value)
            }
            FMIndexEnum::Ucs2(fm_index) => {
                let value = fm_index.suffix_idx(k)?;
                Ok(value)
            }
            FMIndexEnum::Ucs4(fm_index) => {
                let value = fm_index.suffix_idx(k)?;
                Ok(value)
            }
        }
    }

    pub(crate) fn len(&self) -> PyResult<usize> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => fm_index.len(),
            FMIndexEnum::Ucs2(fm_index) => fm_index.len(),
            FMIndexEnum::Ucs4(fm_index) => fm_index.len(),
        }
    }

    pub(crate) fn max_bit(&self) -> PyResult<usize> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => fm_index.max_bit(),
            FMIndexEnum::Ucs2(fm_index) => fm_index.max_bit(),
            FMIndexEnum::Ucs4(fm_index) => fm_index.max_bit(),
        }
    }

    pub(crate) fn value(&self) -> PyResult<String> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => {
                let values = fm_index.values()?;
                String::from_utf8(values).map_err(PyUnicodeDecodeError::new_err)
            }
            FMIndexEnum::Ucs2(fm_index) => {
                let values = fm_index.values()?;
                String::from_utf16(&values).map_err(PyUnicodeDecodeError::new_err)
            }
            FMIndexEnum::Ucs4(fm_index) => {
                let values = fm_index.values()?;
                Ok(values
                    .into_iter()
                    .map(|c| std::char::from_u32(c).unwrap_or('\u{FFFD}'))
                    .collect())
            }
        }
    }

    pub(crate) fn contains(&self, pattern: PyStringData) -> PyResult<bool> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => fm_index.contains(pattern.to_vec()),
                _ => Ok(false),
            },
            FMIndexEnum::Ucs2(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.contains(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => fm_index.contains(pattern.to_vec()),
                _ => Ok(false),
            },
            FMIndexEnum::Ucs4(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.contains(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    fm_index.contains(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => fm_index.contains(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn count(&self, pattern: PyStringData) -> PyResult<usize> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => fm_index.count(pattern.to_vec()),
                _ => Ok(0),
            },
            FMIndexEnum::Ucs2(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.count(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => fm_index.count(pattern.to_vec()),
                _ => Ok(0),
            },
            FMIndexEnum::Ucs4(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.count(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    fm_index.count(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => fm_index.count(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn locate(&self, pattern: PyStringData) -> PyResult<Vec<usize>> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => fm_index.locate(pattern.to_vec()),
                _ => Ok(vec![]),
            },
            FMIndexEnum::Ucs2(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.locate(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => fm_index.locate(pattern.to_vec()),
                _ => Ok(vec![]),
            },
            FMIndexEnum::Ucs4(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.locate(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    fm_index.locate(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => fm_index.locate(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn starts_with(&self, pattern: PyStringData) -> PyResult<bool> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => fm_index.starts_with(pattern.to_vec()),
                _ => Ok(false),
            },
            FMIndexEnum::Ucs2(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.starts_with(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => fm_index.starts_with(pattern.to_vec()),
                _ => Ok(false),
            },
            FMIndexEnum::Ucs4(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.starts_with(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    fm_index.starts_with(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => fm_index.starts_with(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn ends_with(&self, pattern: PyStringData) -> PyResult<bool> {
        match self {
            FMIndexEnum::Ucs1(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => fm_index.ends_with(pattern.to_vec()),
                _ => Ok(false),
            },
            FMIndexEnum::Ucs2(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.ends_with(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => fm_index.ends_with(pattern.to_vec()),
                _ => Ok(false),
            },
            FMIndexEnum::Ucs4(fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    fm_index.ends_with(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    fm_index.ends_with(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => fm_index.ends_with(pattern.to_vec()),
            },
        }
    }
}
