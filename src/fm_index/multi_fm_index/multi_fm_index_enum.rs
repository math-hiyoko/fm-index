use std::{collections, sync};

use pyo3::{
    Bound, PyResult,
    exceptions::{PyTypeError, PyUnicodeDecodeError},
    types::{PyAnyMethods, PySequence, PyString, PyStringData, PyStringMethods},
};
use rayon::prelude::*;

use super::{multi_fm_index::MultiFMIndex, string_data::StringData};

#[derive(Clone)]
pub(crate) enum MultiFMIndexEnum {
    Ucs1(sync::Arc<MultiFMIndex<u8>>),
    Ucs2(sync::Arc<MultiFMIndex<u16>>),
    Ucs4(sync::Arc<MultiFMIndex<u32>>),
}

impl MultiFMIndexEnum {
    pub(crate) fn new(data: Vec<StringData>) -> PyResult<Self> {
        match data.par_iter().max().unwrap_or(&StringData::Ucs1(vec![])) {
            StringData::Ucs1(_) => {
                let data = data
                    .into_iter()
                    .map(|item| match item {
                        StringData::Ucs1(data) => data,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                let multi_fm_index = MultiFMIndex::new(data)?;
                Ok(MultiFMIndexEnum::Ucs1(sync::Arc::new(multi_fm_index)))
            }
            StringData::Ucs2(_) => {
                let data = data
                    .into_iter()
                    .map(|item| match item {
                        StringData::Ucs1(data) => {
                            data.into_iter().map(|c| c as u16).collect::<Vec<_>>()
                        }
                        StringData::Ucs2(data) => data,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                let multi_fm_index = MultiFMIndex::new(data)?;
                Ok(MultiFMIndexEnum::Ucs2(sync::Arc::new(multi_fm_index)))
            }
            StringData::Ucs4(_) => {
                let data = data
                    .into_iter()
                    .map(|item| match item {
                        StringData::Ucs1(data) => {
                            data.into_iter().map(|c| c as u32).collect::<Vec<_>>()
                        }
                        StringData::Ucs2(data) => {
                            data.into_iter().map(|c| c as u32).collect::<Vec<_>>()
                        }
                        StringData::Ucs4(data) => data,
                    })
                    .collect::<Vec<_>>();
                let multi_fm_index = MultiFMIndex::new(data)?;
                Ok(MultiFMIndexEnum::Ucs4(sync::Arc::new(multi_fm_index)))
            }
        }
    }

    pub(super) fn range_search(&self, pattern: PyStringData) -> PyResult<(usize, usize)> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => multi_fm_index.range_search(pattern.to_vec()),
                _ => Ok((0, 0)),
            },
            MultiFMIndexEnum::Ucs2(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.range_search(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => multi_fm_index.range_search(pattern.to_vec()),
                _ => Ok((0, 0)),
            },
            MultiFMIndexEnum::Ucs4(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.range_search(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    multi_fm_index.range_search(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => multi_fm_index.range_search(pattern.to_vec()),
            },
        }
    }

    pub(super) fn doc_offset(&self, k: usize) -> PyResult<(usize, usize)> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => multi_fm_index.doc_offset(k),
            MultiFMIndexEnum::Ucs2(multi_fm_index) => multi_fm_index.doc_offset(k),
            MultiFMIndexEnum::Ucs4(multi_fm_index) => multi_fm_index.doc_offset(k),
        }
    }

    pub(crate) fn len(&self) -> PyResult<usize> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => multi_fm_index.len(),
            MultiFMIndexEnum::Ucs2(multi_fm_index) => multi_fm_index.len(),
            MultiFMIndexEnum::Ucs4(multi_fm_index) => multi_fm_index.len(),
        }
    }

    pub(crate) fn total_num_chars(&self) -> PyResult<usize> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => multi_fm_index.total_num_chars(),
            MultiFMIndexEnum::Ucs2(multi_fm_index) => multi_fm_index.total_num_chars(),
            MultiFMIndexEnum::Ucs4(multi_fm_index) => multi_fm_index.total_num_chars(),
        }
    }

    pub(crate) fn max_bit(&self) -> PyResult<usize> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => multi_fm_index.max_bit(),
            MultiFMIndexEnum::Ucs2(multi_fm_index) => multi_fm_index.max_bit(),
            MultiFMIndexEnum::Ucs4(multi_fm_index) => multi_fm_index.max_bit(),
        }
    }

    pub(crate) fn code_unit(&self) -> &str {
        match self {
            MultiFMIndexEnum::Ucs1(_) => "ucs1",
            MultiFMIndexEnum::Ucs2(_) => "ucs2",
            MultiFMIndexEnum::Ucs4(_) => "ucs4",
        }
    }

    pub(crate) fn values(&self) -> PyResult<Vec<String>> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .into_par_iter()
                    .map(|value| String::from_utf8(value).map_err(PyUnicodeDecodeError::new_err))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(str_list)
            }
            MultiFMIndexEnum::Ucs2(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .into_par_iter()
                    .map(|value| String::from_utf16(&value).map_err(PyUnicodeDecodeError::new_err))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(str_list)
            }
            MultiFMIndexEnum::Ucs4(multi_fm_index) => {
                let str_list = multi_fm_index
                    .values()?
                    .into_par_iter()
                    .map(|value| {
                        Ok(value
                            .into_iter()
                            .map(|c| std::char::from_u32(c).unwrap_or('\u{FFFD}'))
                            .collect::<String>())
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(str_list)
            }
        }
    }

    pub(crate) fn contains(&self, pattern: PyStringData) -> PyResult<bool> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => multi_fm_index.contains(pattern.to_vec()),
                _ => Ok(false),
            },
            MultiFMIndexEnum::Ucs2(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.contains(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => multi_fm_index.contains(pattern.to_vec()),
                _ => Ok(false),
            },
            MultiFMIndexEnum::Ucs4(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.contains(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    multi_fm_index.contains(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => multi_fm_index.contains(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn count_all(&self, pattern: PyStringData) -> PyResult<usize> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => multi_fm_index.count_all(pattern.to_vec()),
                _ => Ok(0),
            },
            MultiFMIndexEnum::Ucs2(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.count_all(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => multi_fm_index.count_all(pattern.to_vec()),
                _ => Ok(0),
            },
            MultiFMIndexEnum::Ucs4(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.count_all(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    multi_fm_index.count_all(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => multi_fm_index.count_all(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn count(
        &self,
        pattern: PyStringData,
    ) -> PyResult<collections::HashMap<usize, usize>> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => multi_fm_index.count(pattern.to_vec()),
                _ => Ok(collections::HashMap::new()),
            },
            MultiFMIndexEnum::Ucs2(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.count(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => multi_fm_index.count(pattern.to_vec()),
                _ => Ok(collections::HashMap::new()),
            },
            MultiFMIndexEnum::Ucs4(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.count(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    multi_fm_index.count(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => multi_fm_index.count(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn locate(
        &self,
        pattern: PyStringData,
    ) -> PyResult<collections::HashMap<usize, Vec<usize>>> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => multi_fm_index.locate(pattern.to_vec()),
                _ => Ok(collections::HashMap::new()),
            },
            MultiFMIndexEnum::Ucs2(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.locate(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => multi_fm_index.locate(pattern.to_vec()),
                _ => Ok(collections::HashMap::new()),
            },
            MultiFMIndexEnum::Ucs4(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.locate(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    multi_fm_index.locate(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => multi_fm_index.locate(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn starts_with(&self, pattern: PyStringData) -> PyResult<Vec<usize>> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => multi_fm_index.starts_with(pattern.to_vec()),
                _ => Ok(vec![]),
            },
            MultiFMIndexEnum::Ucs2(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.starts_with(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => multi_fm_index.starts_with(pattern.to_vec()),
                _ => Ok(vec![]),
            },
            MultiFMIndexEnum::Ucs4(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.starts_with(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    multi_fm_index.starts_with(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => multi_fm_index.starts_with(pattern.to_vec()),
            },
        }
    }

    pub(crate) fn ends_with(&self, pattern: PyStringData) -> PyResult<Vec<usize>> {
        match self {
            MultiFMIndexEnum::Ucs1(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => multi_fm_index.ends_with(pattern.to_vec()),
                _ => Ok(vec![]),
            },
            MultiFMIndexEnum::Ucs2(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.ends_with(pattern.iter().map(|&c| c as u16).collect())
                }
                PyStringData::Ucs2(pattern) => multi_fm_index.ends_with(pattern.to_vec()),
                _ => Ok(vec![]),
            },
            MultiFMIndexEnum::Ucs4(multi_fm_index) => match pattern {
                PyStringData::Ucs1(pattern) => {
                    multi_fm_index.ends_with(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs2(pattern) => {
                    multi_fm_index.ends_with(pattern.iter().map(|&c| c as u32).collect())
                }
                PyStringData::Ucs4(pattern) => multi_fm_index.ends_with(pattern.to_vec()),
            },
        }
    }
}
