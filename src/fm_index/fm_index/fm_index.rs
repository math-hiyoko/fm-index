use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::{
    fm_index::{base_fm_index::base_fm_index::BaseFMIndex, traits::fm_index::FMIndexTrait},
    utils::suffix_array::suffix_array_vec,
};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct FMIndex {
    base_fm_index: BaseFMIndex,
}

impl FMIndex {
    pub(crate) fn new(data: Vec<u32>) -> PyResult<Self> {
        let suffix_idx = suffix_array_vec(&data)?;
        let base_fm_index = BaseFMIndex::new(data, suffix_idx)?;
        Ok(FMIndex { base_fm_index })
    }
}

impl FMIndexTrait for FMIndex {
    type BaseFMIndex = BaseFMIndex;

    fn get_base_fm_index(&self) -> &Self::BaseFMIndex {
        &self.base_fm_index
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use num_traits::Zero;

    use super::*;

    fn create_fm_index(data: &str) -> PyResult<FMIndex> {
        let data = data
            .chars()
            .map(|c| c as u32 + 1)
            .chain(iter::once(0))
            .collect();
        FMIndex::new(data)
    }

    #[test]
    fn test_empty_index() {
        let data = "";
        let index = create_fm_index(data).unwrap();

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
        let index = create_fm_index(data).unwrap();

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
        let index = create_fm_index(data).unwrap();

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
        let index = create_fm_index(text).unwrap();

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
