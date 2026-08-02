use std::{collections, fs, mem};

use bytemuck::{cast_slice, cast_slice_mut};
use memmap2::{Mmap, MmapMut};
use pyo3::{PyResult, exceptions::PyOSError};
use rayon::prelude::*;
use tempfile::tempfile;

use crate::{
    fm_index::traits::base_fm_index::{ARRAY_SAMPLING_RATE, BaseFMIndexTrait},
    utils::disk_wavelet_matrix::{
        disk_bit_vector::DiskBitVector, disk_wavelet_matrix::DiskWaveletMatrix,
    },
};

pub(crate) struct DiskBaseFMIndex {
    len: usize,
    zero_suffix_idx: usize,
    suffix_idx_sampled_mmap: Mmap,
    _suffix_idx_sampled_file: fs::File,
    counts_less: collections::HashMap<u32, usize>,
    burrows_wheeler_transform: DiskWaveletMatrix<u32>,
}

impl DiskBaseFMIndex {
    pub(in crate::fm_index) fn new(data: Mmap, suffix_idx: Mmap) -> PyResult<Self> {
        let len = data.len() / mem::size_of::<u32>();
        let data_slice = cast_slice::<u8, u32>(&data);
        let suffix_idx = cast_slice::<u8, usize>(&suffix_idx);

        let zero_suffix_idx = suffix_idx
            .par_iter()
            .position_any(|&idx| idx == 0)
            .unwrap_or(0usize);

        let suffix_idx_sampled_file = tempfile().map_err(PyOSError::new_err)?;
        suffix_idx_sampled_file
            .set_len(
                (suffix_idx.len().div_ceil(ARRAY_SAMPLING_RATE) * mem::size_of::<usize>()) as u64,
            )
            .map_err(PyOSError::new_err)?;
        #[allow(unsafe_code)]
        let mut suffix_idx_sampled =
            unsafe { MmapMut::map_mut(&suffix_idx_sampled_file).map_err(PyOSError::new_err)? };
        let suffix_idx_sampled_slice = cast_slice_mut::<u8, usize>(&mut suffix_idx_sampled);
        suffix_idx_sampled_slice.copy_from_slice(
            &suffix_idx
                .iter()
                .step_by(ARRAY_SAMPLING_RATE)
                .copied()
                .collect::<Vec<_>>(),
        );

        let mut counts_less = collections::HashMap::new();
        for (cumulative_count, &idx) in suffix_idx.iter().enumerate() {
            let symbol = data_slice[idx];
            counts_less.entry(symbol).or_insert(cumulative_count);
        }

        let burrows_wheeler_transform_file = tempfile().map_err(PyOSError::new_err)?;
        burrows_wheeler_transform_file
            .set_len((len * mem::size_of::<u32>()) as u64)
            .map_err(PyOSError::new_err)?;
        #[allow(unsafe_code)]
        let mut burrows_wheeler_transform = unsafe {
            MmapMut::map_mut(&burrows_wheeler_transform_file).map_err(PyOSError::new_err)?
        };
        let burrows_wheeler_transform_slice =
            cast_slice_mut::<u8, u32>(&mut burrows_wheeler_transform);
        burrows_wheeler_transform_slice.copy_from_slice(
            &suffix_idx
                .par_iter()
                .map(|&idx| {
                    if idx == 0 {
                        data_slice[len - 1]
                    } else {
                        data_slice[idx - 1]
                    }
                })
                .collect::<Vec<_>>(),
        );
        let burrows_wheeler_transform =
            DiskWaveletMatrix::new(burrows_wheeler_transform, burrows_wheeler_transform_file)?;

        Ok(Self {
            len,
            zero_suffix_idx,
            suffix_idx_sampled_mmap: suffix_idx_sampled
                .make_read_only()
                .map_err(PyOSError::new_err)?,
            _suffix_idx_sampled_file: suffix_idx_sampled_file,
            counts_less,
            burrows_wheeler_transform,
        })
    }

    pub(in crate::fm_index) fn try_clone(&self) -> PyResult<Self> {
        let suffix_idx_sampled_file = tempfile().map_err(PyOSError::new_err)?;
        suffix_idx_sampled_file
            .set_len(self.suffix_idx_sampled_mmap.len() as u64)
            .map_err(PyOSError::new_err)?;
        #[allow(unsafe_code)]
        let mut suffix_idx_sampled_mmapmut =
            unsafe { MmapMut::map_mut(&suffix_idx_sampled_file).map_err(PyOSError::new_err)? };
        suffix_idx_sampled_mmapmut.copy_from_slice(&self.suffix_idx_sampled_mmap[..]);
        let suffix_idx_sampled_mmap = suffix_idx_sampled_mmapmut
            .make_read_only()
            .map_err(PyOSError::new_err)?;

        Ok(Self {
            len: self.len,
            zero_suffix_idx: self.zero_suffix_idx,
            suffix_idx_sampled_mmap,
            _suffix_idx_sampled_file: suffix_idx_sampled_file,
            counts_less: self.counts_less.clone(),
            burrows_wheeler_transform: self.burrows_wheeler_transform.try_clone()?,
        })
    }
}

impl BaseFMIndexTrait for DiskBaseFMIndex {
    type BitVector = DiskBitVector;
    type WaveletMatrix = DiskWaveletMatrix<u32>;

    fn len(&self) -> usize {
        self.len
    }

    fn get_zero_suffix_idx(&self) -> usize {
        self.zero_suffix_idx
    }

    fn get_suffix_idx_sampled(&self) -> &[usize] {
        cast_slice::<u8, usize>(&self.suffix_idx_sampled_mmap)
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
    use crate::utils::suffix_array::{suffix_array_mmap, suffix_array_vec};

    #[test]
    fn test_base_fm_index_empty() {
        let raw_slice = vec![0u32];
        let mut mmap = MmapMut::map_anon(raw_slice.len() * mem::size_of::<u32>()).unwrap();
        {
            let slice: &mut [u32] = cast_slice_mut(&mut mmap);
            slice.copy_from_slice(&raw_slice);
        }
        let data_mmap = mmap.make_read_only().unwrap();

        let (suffix_idx, _) = suffix_array_mmap(&data_mmap).unwrap();

        let fm_index = DiskBaseFMIndex::new(data_mmap, suffix_idx).unwrap();

        assert_eq!(fm_index.suffix_idx(0).unwrap(), 0);
        assert_eq!(fm_index.values().unwrap(), vec![0]);
        assert_eq!(fm_index.range_search(vec![0]).unwrap(), (0, 1));
    }

    #[test]
    fn test_base_fm_index_single_char() {
        let raw_data = "aaaaaaaaaa"
            .chars()
            .map(|c| c as u32)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let mut data_mmap = MmapMut::map_anon(raw_data.len() * mem::size_of::<u32>()).unwrap();
        {
            let slice: &mut [u32] = cast_slice_mut(&mut data_mmap);
            slice.copy_from_slice(&raw_data);
        }
        let data_mmap = data_mmap.make_read_only().unwrap();

        let (suffix_idx, _) = suffix_array_mmap(&data_mmap).unwrap();
        let fm_index = DiskBaseFMIndex::new(data_mmap, suffix_idx).unwrap();

        for (i, &suffix_idx) in suffix_array_vec(&raw_data)
            .unwrap()
            .iter()
            .enumerate()
            .take(raw_data.len())
        {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), raw_data);
        assert_eq!(
            fm_index
                .range_search(vec![b'a' as u32, b'a' as u32, b'a' as u32])
                .unwrap(),
            (3, 11),
        );
    }

    #[test]
    fn test_base_fm_index_u32() {
        let raw_data = "にわにはにわにわとりがいる"
            .chars()
            .map(|c| c as u32)
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        let mut data_mmap = MmapMut::map_anon(raw_data.len() * mem::size_of::<u32>()).unwrap();
        {
            let slice: &mut [u32] = cast_slice_mut(&mut data_mmap);
            slice.copy_from_slice(&raw_data);
        }
        let data_mmap = data_mmap.make_read_only().unwrap();

        let (suffix_idx, _) = suffix_array_mmap(&data_mmap).unwrap();
        let fm_index = DiskBaseFMIndex::new(data_mmap, suffix_idx).unwrap();

        for (i, &suffix_idx) in suffix_array_vec(&raw_data)
            .unwrap()
            .iter()
            .enumerate()
            .take(raw_data.len())
        {
            assert_eq!(fm_index.suffix_idx(i).unwrap(), suffix_idx);
        }
        assert_eq!(fm_index.values().unwrap(), raw_data);
        assert_eq!(
            fm_index
                .range_search(vec!['に' as u32, 'わ' as u32])
                .unwrap(),
            (5, 8),
        );
    }
}
