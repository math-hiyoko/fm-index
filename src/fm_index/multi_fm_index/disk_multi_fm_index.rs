use std::{fs, iter, mem};

use bytemuck::{cast_slice, cast_slice_mut};
use memmap2::{Mmap, MmapMut};
use num_traits::Zero;
use pyo3::{PyResult, exceptions::PyOSError};
use tempfile::tempfile;

use crate::{
    fm_index::{
        base_fm_index::disk_base_fm_index::DiskBaseFMIndex,
        traits::multi_fm_index::MultiFMIndexTrait,
    },
    utils::{
        disk_wavelet_matrix::{
            disk_bit_vector::DiskBitVector, disk_wavelet_matrix::DiskWaveletMatrix,
        },
        suffix_array::suffix_array_mmap,
    },
};

pub(crate) struct DiskMultiFMIndex {
    base_fm_index: DiskBaseFMIndex,
    doc_start_index_mmap: Mmap,
    doc_start_index_file: fs::File,
    doc_id_of_index: DiskWaveletMatrix<usize>,
}

impl DiskMultiFMIndex {
    pub(crate) fn new(data: Mmap) -> PyResult<Self> {
        let data_slice = cast_slice::<u8, u32>(&data);

        let (suffix_idx_mmap, _) = suffix_array_mmap(&data)?;

        let doc_id_of_index = {
            let doc_ids_file = tempfile().map_err(PyOSError::new_err)?;
            doc_ids_file
                .set_len((data.len() / mem::size_of::<u32>()) as u64)
                .map_err(PyOSError::new_err)?;
            #[allow(unsafe_code)]
            let mut doc_ids_mmap =
                unsafe { MmapMut::map_mut(&doc_ids_file).map_err(PyOSError::new_err)? };
            let doc_ids_slice: &mut [usize] = cast_slice_mut(&mut doc_ids_mmap[..]);
            data_slice
                .iter()
                .scan(0usize, |doc_id, &value| {
                    let ret = *doc_id;
                    if value.is_zero() {
                        *doc_id += 1;
                    }
                    Some(ret)
                })
                .enumerate()
                .for_each(|(i, doc_id)| {
                    doc_ids_slice[i] = doc_id;
                });

            let doc_ids_of_suffix_idx_file = tempfile().map_err(PyOSError::new_err)?;
            doc_ids_of_suffix_idx_file
                .set_len((suffix_idx_mmap.len() / mem::size_of::<usize>()) as u64)
                .map_err(PyOSError::new_err)?;
            #[allow(unsafe_code)]
            let mut doc_ids_of_suffix_idx_mmap = unsafe {
                MmapMut::map_mut(&doc_ids_of_suffix_idx_file).map_err(PyOSError::new_err)?
            };
            let doc_ids_of_suffix_idx_slice: &mut [usize] =
                cast_slice_mut(&mut doc_ids_of_suffix_idx_mmap[..]);
            let suffix_idx_slice = cast_slice::<u8, usize>(&suffix_idx_mmap);
            suffix_idx_slice
                .iter()
                .map(|&idx| doc_ids_slice[idx])
                .enumerate()
                .for_each(|(i, doc_id)| {
                    doc_ids_of_suffix_idx_slice[i] = doc_id;
                });

            DiskWaveletMatrix::<usize>::new(doc_ids_of_suffix_idx_mmap, doc_ids_of_suffix_idx_file).map_err(PyOSError::new_err)?
        };

        let doc_start_index_file = tempfile().map_err(PyOSError::new_err)?;
        doc_start_index_file
            .set_len(
                (data_slice.iter().filter(|&&c| c.is_zero()).count() * mem::size_of::<usize>())
                    as u64,
            )
            .map_err(PyOSError::new_err)?;
        #[allow(unsafe_code)]
        let mut doc_start_index_mmap =
            unsafe { MmapMut::map_mut(&doc_start_index_file).map_err(PyOSError::new_err)? };
        let doc_start_index_slice: &mut [usize] = cast_slice_mut(&mut doc_start_index_mmap[..]);
        iter::once(0)
            .chain(data_slice.iter().enumerate().filter_map(|(i, &value)| {
                if value.is_zero() && i + 1 < data_slice.len() {
                    Some(i)
                } else {
                    None
                }
            }))
            .enumerate()
            .for_each(|(i, value)| {
                doc_start_index_slice[i] = value;
            });

        let base_fm_index = DiskBaseFMIndex::new(data, suffix_idx_mmap)?;

        Ok(DiskMultiFMIndex {
            base_fm_index,
            doc_start_index_mmap: doc_start_index_mmap
                .make_read_only()
                .map_err(PyOSError::new_err)?,
            doc_start_index_file,
            doc_id_of_index,
        })
    }

    pub(crate) fn try_clone(&self) -> PyResult<Self> {
        let doc_start_index_file = tempfile().map_err(PyOSError::new_err)?;
        doc_start_index_file
            .set_len(self.doc_start_index_mmap.len() as u64)
            .map_err(PyOSError::new_err)?;
        #[allow(unsafe_code)]
        let mut doc_start_index_mmap =
            unsafe { MmapMut::map_mut(&doc_start_index_file).map_err(PyOSError::new_err)? };
        doc_start_index_mmap.copy_from_slice(&self.doc_start_index_mmap[..]);
        Ok(Self {
            base_fm_index: self.base_fm_index.try_clone()?,
            doc_start_index_mmap: doc_start_index_mmap
                .make_read_only()
                .map_err(PyOSError::new_err)?,
            doc_start_index_file,
            doc_id_of_index: self.doc_id_of_index.try_clone()?,
        })
    }
}

impl MultiFMIndexTrait for DiskMultiFMIndex {
    type BitVector = DiskBitVector;
    type WaveletMatrix = DiskWaveletMatrix<usize>;
    type BaseFMIndex = DiskBaseFMIndex;

    fn get_num_docs(&self) -> usize {
        self.doc_start_index_mmap.len() / mem::size_of::<usize>()
    }

    fn get_base_fm_index(&self) -> &Self::BaseFMIndex {
        &self.base_fm_index
    }

    fn get_doc_start_index(&self) -> &[usize] {
        cast_slice(&self.doc_start_index_mmap[..])
    }

    fn get_doc_id_of_index(&self) -> &Self::WaveletMatrix {
        &self.doc_id_of_index
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use pyo3::Python;
    use std::collections;

    use super::*;

    fn create_multi_fm_index(data: Vec<String>) -> DiskMultiFMIndex {
        let data = data
            .iter()
            .flat_map(|doc| doc.chars().map(|c| c as u32 + 1).chain(iter::once(0)))
            .collect::<Vec<_>>();
        let mut data_mmap = MmapMut::map_anon(data.len() * std::mem::size_of::<u32>()).unwrap();
        let data_slice: &mut [u32] = cast_slice_mut(&mut data_mmap[..]);
        data_slice.copy_from_slice(&data[..]);
        DiskMultiFMIndex::new(data_mmap.make_read_only().unwrap()).unwrap()
    }

    #[test]
    fn test_empty_collection() {
        let data = Vec::<String>::new();
        let index = create_multi_fm_index(data);

        // Length and values
        assert!(index.get_num_docs().is_zero());
        assert!(index.values().unwrap().is_empty());

        // Contains and count
        assert!(!index.contains("").unwrap());
        assert!(!index.contains("a").unwrap());
        assert!(index.count_all("").unwrap().is_zero());
        assert!(index.count_all("a").unwrap().is_zero());
        assert!(index.count("").unwrap().is_empty());
        assert!(index.count("a").unwrap().is_empty());

        // Locate
        assert!(index.locate("").unwrap().is_empty());
        assert!(index.locate("a").unwrap().is_empty());

        // Starts with and ends with
        assert!(index.starts_with("").unwrap().is_empty());
        assert!(index.starts_with("a").unwrap().is_empty());
        assert!(index.ends_with("").unwrap().is_empty());
        assert!(index.ends_with("a").unwrap().is_empty());
    }

    #[test]
    fn test_collection_of_empty_documents() {
        let data = vec!["".to_string(), "".to_string(), "".to_string()];
        let index = create_multi_fm_index(data);

        let expected_values: Vec<String> = vec!["".to_string(), "".to_string(), "".to_string()];

        // Length and values
        assert_eq!(index.get_num_docs(), 3);
        assert_eq!(index.values().unwrap(), expected_values);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(!index.contains("a").unwrap());
        assert_eq!(index.count_all("").unwrap(), 3);
        assert_eq!(index.count_all("a").unwrap(), 0);
        assert_eq!(
            index.count("").unwrap(),
            collections::HashMap::from([(0, 1), (1, 1), (2, 1)])
        );
        assert!(index.count("a").unwrap().is_empty());
        assert_eq!(index.count_within_doc(1, "").unwrap(), 1,);
        assert_eq!(index.count_within_doc(1, "a").unwrap(), 0,);

        // Locate
        assert_eq!(
            index.locate("").unwrap(),
            collections::HashMap::from([(0, vec![0]), (1, vec![0]), (2, vec![0])])
        );
        assert!(index.locate("a").unwrap().is_empty());
        assert_eq!(index.locate_within_doc(1, "").unwrap(), vec![0],);
        assert!(index.locate_within_doc(1, "a").unwrap().is_empty());

        // Starts with and ends with
        assert_eq!(index.starts_with("").unwrap(), [2, 1, 0]);
        assert!(index.starts_with("a").unwrap().is_empty());
        assert_eq!(index.ends_with("").unwrap(), [2, 1, 0]);
        assert!(index.ends_with("a").unwrap().is_empty());
    }

    #[test]
    fn test_single_repeated_character_documents() {
        let data = vec![
            "aaaaaaaaaa".to_string(),
            "".to_string(),
            "aaaaaa".to_string(),
            "aaaaaaaa".to_string(),
        ];
        let index = create_multi_fm_index(data);

        let expected_values = vec![
            "aaaaaaaaaa".to_string(),
            "".to_string(),
            "aaaaaa".to_string(),
            "aaaaaaaa".to_string(),
        ];

        // Length and values
        assert_eq!(index.get_num_docs(), 4);
        assert_eq!(index.values().unwrap(), expected_values);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(!index.contains("a").unwrap());
        assert!(index.contains("aaaaaa").unwrap());
        assert_eq!(index.count_all("").unwrap(), 28);
        assert_eq!(index.count_all("aa").unwrap(), 21);
        assert_eq!(
            index.count("").unwrap(),
            collections::HashMap::from([(0, 11), (1, 1), (2, 7), (3, 9)])
        );
        assert_eq!(
            index.count("aa").unwrap(),
            collections::HashMap::from([(0, 9), (2, 5), (3, 7)])
        );
        assert_eq!(index.count_within_doc(0, "").unwrap(), 11,);
        assert_eq!(index.count_within_doc(0, "aa").unwrap(), 9,);

        // Locate
        assert_eq!(
            index.locate("").unwrap(),
            collections::HashMap::from([
                (0, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
                (1, vec![0]),
                (2, vec![6, 5, 4, 3, 2, 1, 0]),
                (3, vec![8, 7, 6, 5, 4, 3, 2, 1, 0])
            ])
        );
        assert_eq!(
            index.locate("aa").unwrap(),
            collections::HashMap::from([
                (0, vec![8, 7, 6, 5, 4, 3, 2, 1, 0]),
                (2, vec![4, 3, 2, 1, 0]),
                (3, vec![6, 5, 4, 3, 2, 1, 0])
            ])
        );
        assert_eq!(
            index.locate_within_doc(0, "").unwrap(),
            vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        );
        assert_eq!(
            index.locate_within_doc(0, "aa").unwrap(),
            vec![8, 7, 6, 5, 4, 3, 2, 1, 0],
        );

        // Starts with and ends with
        assert_eq!(index.starts_with("").unwrap(), [1, 2, 3, 0]);
        assert_eq!(index.starts_with("aa").unwrap(), [2, 3, 0]);
        assert_eq!(index.ends_with("").unwrap(), [3, 0, 1, 2]);
        assert_eq!(index.ends_with("aa").unwrap(), [3, 0, 2]);
    }

    #[test]
    fn test_multiple_byte_string_documents() {
        let data = vec![
            "banana".to_string(),
            "bandana".to_string(),
            "anaba".to_string(),
        ];
        let index = create_multi_fm_index(data);

        let expected_values = vec![
            "banana".to_string(),
            "bandana".to_string(),
            "anaba".to_string(),
        ];

        // Length and values
        assert_eq!(index.get_num_docs(), 3);
        assert_eq!(index.values().unwrap(), expected_values);

        // Contains and count
        assert!(!index.contains("").unwrap());
        assert!(!index.contains("ana").unwrap());
        assert!(index.contains("banana").unwrap());
        assert_eq!(index.count_all("ana").unwrap(), 4);
        assert_eq!(
            index.count("ana").unwrap(),
            collections::HashMap::from([(0, 2), (1, 1), (2, 1)])
        );
        assert_eq!(index.count_within_doc(1, "ana").unwrap(), 1,);

        // Locate
        assert_eq!(
            index.locate("ana").unwrap(),
            collections::HashMap::from([(0, vec![3, 1]), (1, vec![4]), (2, vec![0])])
        );
        assert_eq!(index.locate_within_doc(1, "ana").unwrap(), vec![4],);

        // Starts with and ends with
        assert_eq!(index.starts_with("ba").unwrap(), [0, 1]);
        assert_eq!(index.ends_with("na").unwrap(), [1, 0]);
    }

    #[test]
    fn test_topk_basic() {
        let data = vec![
            "abcabcabcabc".to_string(),
            "xxabcabcxxabc".to_string(),
            "abcababcabc".to_string(),
        ];
        let index = create_multi_fm_index(data);

        // Get top 2 documents with "abc"
        let result = index.topk("abc", 2).unwrap();
        assert_eq!(result, vec![(0, 4), (1, 3)]);

        // Get top 3 documents with "abc" (all 3 documents have matches)
        let result = index.topk("abc", 3).unwrap();
        assert_eq!(result, vec![(0, 4), (1, 3), (2, 3)]);

        // Get top 5 documents with "abc" (only 3 documents exist)
        let result = index.topk("abc", 5).unwrap();
        assert_eq!(result, vec![(0, 4), (1, 3), (2, 3)]);

        // Get top 1 document with "abc"
        let result = index.topk("abc", 1).unwrap();
        assert_eq!(result, vec![(0, 4)]);
    }

    #[test]
    fn test_topk_no_matches() {
        let data = vec![
            "banana".to_string(),
            "bandana".to_string(),
            "anaba".to_string(),
        ];
        let index = create_multi_fm_index(data);

        // Pattern not found in any document
        let result = index.topk("xyz", 2).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_topk_single_match() {
        let data = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello world".to_string(),
        ];
        let index = create_multi_fm_index(data);

        // Pattern "hello" appears in docs 0 and 2
        let mut result = index.topk("hello", 2).unwrap();
        // Sort by doc_id to ensure consistent ordering
        result.sort_by_key(|(doc_id, _)| *doc_id);
        assert_eq!(result, vec![(0, 1), (2, 1)]);
    }

    #[test]
    fn test_topk_different_counts() {
        let data = vec![
            "aaaaaaaaaa".to_string(),
            "aaa".to_string(),
            "aaaa".to_string(),
            "aa".to_string(),
        ];
        let index = create_multi_fm_index(data);

        // Get top 3 documents with "aa"
        let result = index.topk("aa", 3).unwrap();
        assert_eq!(result, vec![(0, 9), (2, 3), (1, 2)]);
    }

    #[test]
    fn test_topk_empty_collection() {
        let data = Vec::<String>::new();
        let index = create_multi_fm_index(data);

        // Should return error for k > 0 with empty collection
        let result = index.topk("a", 1).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_topk_k_zero() {
        Python::initialize();

        let data = vec!["abc".to_string(), "def".to_string()];
        let index = create_multi_fm_index(data);

        // k must be greater than 0
        let result = index.topk("abc", 0);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "ValueError: k must be greater than 0"
        );
    }
}
