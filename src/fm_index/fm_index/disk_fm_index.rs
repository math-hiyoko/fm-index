use memmap2::Mmap;
use pyo3::{PyResult, exceptions::PyOSError};

use crate::{
    fm_index::{
        base_fm_index::disk_base_fm_index::DiskBaseFMIndex, traits::fm_index::FMIndexTrait,
    },
    utils::suffix_array::suffix_array_mmap,
};

pub(crate) struct DiskFMIndex {
    base_fm_index: DiskBaseFMIndex,
}

impl DiskFMIndex {
    pub(crate) fn new(data: Mmap) -> PyResult<Self> {
        let (suffix_idx, _) = suffix_array_mmap(&data).map_err(PyOSError::new_err)?;
        let base_fm_index = DiskBaseFMIndex::new(data, suffix_idx)?;
        Ok(Self { base_fm_index })
    }

    pub(crate) fn try_clone(&self) -> PyResult<Self> {
        Ok(Self {
            base_fm_index: self.base_fm_index.try_clone()?,
        })
    }
}

impl FMIndexTrait for DiskFMIndex {
    type BaseFMIndex = DiskBaseFMIndex;

    fn get_base_fm_index(&self) -> &Self::BaseFMIndex {
        &self.base_fm_index
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::cast_slice_mut;
    use memmap2::MmapMut;
    use num_traits::Zero;
    use tempfile::tempfile;

    use super::*;

    fn create_disk_fm_index(data: &str) -> PyResult<DiskFMIndex> {
        let data_file = tempfile().map_err(PyOSError::new_err)?;
        data_file
            .set_len(((data.chars().count() + 1) * std::mem::size_of::<u32>()) as u64)
            .map_err(PyOSError::new_err)?;
        #[allow(unsafe_code)]
        let mut data_mmap =
            unsafe { MmapMut::map_mut(&data_file).map_err(PyOSError::new_err)? };
        let data_slice = cast_slice_mut::<u8, u32>(&mut data_mmap[..]);
        for (i, c) in data.chars().enumerate() {
            data_slice[i] = c as u32 + 1;
        }
        data_slice[data.chars().count()] = 0; // null terminator
        let data_mmap = data_mmap
            .make_read_only()
            .map_err(PyOSError::new_err)?;
        DiskFMIndex::new(data_mmap)
    }

    #[test]
    fn test_empty_index() {
        let data = "";
        let index = create_disk_fm_index(data).unwrap();

        // Length and values
        assert!(index.len().is_zero());
        assert!(index.value().unwrap().is_empty());

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(!index.contains("a").unwrap());
        assert_eq!(index.count("").unwrap(), 1);
        assert!(index.count("a").unwrap().is_zero());

        // Locate
        assert_eq!(index.locate("").unwrap(), [0]);
        assert!(index.locate("a").unwrap().is_empty());

        // Starts with and ends with
        assert!(index.starts_with("").unwrap());
        assert!(!index.starts_with("a").unwrap());
        assert!(index.ends_with("").unwrap());
        assert!(!index.ends_with("a").unwrap());
    }

    #[test]
    fn test_single_repeated_character() {
        let data = "aaaaaaaaaa";
        let index = create_disk_fm_index(data).unwrap();

        // Length and values
        assert_eq!(index.len(), 10);
        assert_eq!(index.value().unwrap(), data);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(index.contains("a").unwrap());
        assert_eq!(index.count("a").unwrap(), 10);

        // Locate
        assert_eq!(
            {
                let mut sorted = index.locate("a").unwrap();
                sorted.sort();
                sorted
            },
            (0..10).collect::<Vec<_>>()
        );

        // Starts with and ends with
        assert!(index.starts_with("").unwrap());
        assert!(index.starts_with("aa").unwrap());
        assert!(!index.starts_with("bb").unwrap());
        assert!(index.ends_with("").unwrap());
        assert!(index.ends_with("aa").unwrap());
        assert!(!index.ends_with("bb").unwrap());
    }

    #[test]
    fn test_byte_string_operations() {
        let data = "mississippi";
        let index = create_disk_fm_index(data).unwrap();

        // Length and values
        assert_eq!(index.len(), 11);
        assert_eq!(index.value().unwrap(), data);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(index.contains("is").unwrap());
        assert_eq!(index.count("is").unwrap(), 2);

        // Locate
        assert_eq!(index.locate("is").unwrap(), [4, 1]);

        // Starts with
        assert!(index.starts_with("").unwrap());
        assert!(index.starts_with("mi").unwrap());
        assert!(!index.starts_with("si").unwrap());

        // Ends with
        assert!(index.ends_with("").unwrap());
        assert!(index.ends_with("pi").unwrap());
        assert!(!index.ends_with("ip").unwrap());
    }

    #[test]
    fn test_unicode_string_operations() {
        let text = "にわにはにわにわとりがいる";
        let index = create_disk_fm_index(text).unwrap();

        // Length and values
        assert_eq!(index.len(), 13);
        assert_eq!(index.value().unwrap(), text);

        // Contains and count
        assert!(index.contains("").unwrap());
        assert!(index.contains("にわ").unwrap());
        assert_eq!(index.count("にわ").unwrap(), 3);

        // Locate
        assert_eq!(index.locate("にわ").unwrap(), [6, 0, 4]);

        // Starts with
        assert!(index.starts_with("").unwrap());
        assert!(index.starts_with("にわ").unwrap());
        assert!(!index.starts_with("いる").unwrap());

        // Ends with
        assert!(index.ends_with("").unwrap());
        assert!(index.ends_with("いる").unwrap());
        assert!(!index.ends_with("にわ").unwrap());
    }
}
