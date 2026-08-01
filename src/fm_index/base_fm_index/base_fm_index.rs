use std::collections;

use pyo3::PyResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    fm_index::traits::base_fm_index::{ARRAY_SAMPLING_RATE, BaseFMIndexTrait},
    utils::wavelet_matrix::{bit_vector::BitVector, wavelet_matrix::WaveletMatrix},
};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct BaseFMIndex {
    len: usize,
    zero_suffix_idx: usize,
    suffix_idx_sampled: Vec<usize>,
    counts_less: collections::HashMap<u32, usize>,
    burrows_wheeler_transform: WaveletMatrix<u32>,
}

impl BaseFMIndex {
    pub(in crate::fm_index) fn new(data: Vec<u32>, suffix_idx: Vec<usize>) -> PyResult<Self> {
        let len = data.len();

        let zero_suffix_idx = suffix_idx
            .par_iter()
            .position_any(|&idx| idx == 0)
            .unwrap_or(0usize);
        let suffix_idx_sampled = suffix_idx
            .iter()
            .step_by(ARRAY_SAMPLING_RATE)
            .copied()
            .collect::<Vec<_>>();

        let mut counts_less = collections::HashMap::new();
        for (cumulative_count, &idx) in suffix_idx.iter().enumerate() {
            let symbol = data[idx];
            counts_less.entry(symbol).or_insert(cumulative_count);
        }

        let burrows_wheeler_transform = suffix_idx
            .into_par_iter()
            .map(|idx| {
                if idx == 0 {
                    data[len - 1]
                } else {
                    data[idx - 1]
                }
            })
            .collect::<Vec<_>>();
        let burrows_wheeler_transform = WaveletMatrix::new(burrows_wheeler_transform)?;

        Ok(BaseFMIndex {
            len,
            zero_suffix_idx,
            suffix_idx_sampled,
            counts_less,
            burrows_wheeler_transform,
        })
    }
}

impl BaseFMIndexTrait for BaseFMIndex {
    type BitVector = BitVector;
    type WaveletMatrix = WaveletMatrix<u32>;

    fn len(&self) -> usize {
        self.len
    }

    fn get_zero_suffix_idx(&self) -> usize {
        self.zero_suffix_idx
    }

    fn get_suffix_idx_sampled(&self) -> &[usize] {
        &self.suffix_idx_sampled
    }

    fn get_counts_less(&self) -> &collections::HashMap<u32, usize> {
        &self.counts_less
    }

    fn get_burrows_wheeler_transform(&self) -> &Self::WaveletMatrix {
        &self.burrows_wheeler_transform
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::utils::suffix_array::suffix_array_vec;

    #[test]
    fn test_base_fm_index_empty() {
        let data = vec![0];
        let suffix_idx = suffix_array_vec(&data).unwrap();
        let fm_index = BaseFMIndex::new(data, suffix_idx).unwrap();

        assert_eq!(fm_index.suffix_idx(0).unwrap(), 0);
        assert_eq!(fm_index.values().unwrap(), vec![0]);
        assert_eq!(fm_index.range_search(vec![0]).unwrap(), (0, 1));
    }

    #[test]
    fn test_base_fm_index_single_char() {
        let data = "aaaaaaaaaa"
            .chars()
            .map(|c| c as u32)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let suffix_idx = suffix_array_vec(&data).unwrap();
        let fm_index = BaseFMIndex::new(data.to_vec(), suffix_idx.clone()).unwrap();

        for (i, &suffix_idx) in suffix_idx.iter().enumerate().take(data.len()) {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(
            fm_index
                .range_search(vec![b'a' as u32, b'a' as u32, b'a' as u32])
                .unwrap(),
            (3, 11),
        );
    }

    #[test]
    fn test_base_fm_index_u32() {
        let data = "にわにはにわにわとりがいる"
            .chars()
            .map(|c| c as u32)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let suffix_idx = suffix_array_vec(&data).unwrap();
        let fm_index = BaseFMIndex::new(data.clone(), suffix_idx.clone()).unwrap();

        for (i, &suffix_idx) in suffix_idx.iter().enumerate() {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), data);
        assert_eq!(
            fm_index
                .range_search(vec!['に' as u32, 'わ' as u32])
                .unwrap(),
            (5, 8),
        );
    }
}
