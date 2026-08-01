use std::{fs, iter, marker, mem, ops};

use bytemuck::{Pod, cast_slice, cast_slice_mut};
use memmap2::MmapMut;
use num_traits::{One, PrimInt, Unsigned};
use pyo3::{PyResult, exceptions::PyOSError};
use rayon::prelude::*;
use tempfile::tempfile;

use super::disk_bit_vector::DiskBitVector;
use crate::utils::traits::{
    bit_vector::BlockType, bit_width::BitWidth, wavelet_matrix::WaveletMatrixTrait,
};

pub(crate) struct DiskWaveletMatrix<NumberType> {
    layers: Vec<DiskBitVector>,
    zeros_count_per_layer: Vec<usize>,
    height: usize,
    len: usize,
    phantom: marker::PhantomData<NumberType>,
}

impl<NumberType> DiskWaveletMatrix<NumberType>
where
    NumberType: BitWidth + One + PrimInt + Unsigned + Pod + Send + Sync,
{
    pub(crate) fn new(mut values: MmapMut, values_file: fs::File) -> PyResult<Self> {
        assert!(values.len().is_multiple_of(mem::size_of::<NumberType>()));
        let len = values.len() / mem::size_of::<NumberType>();

        let mut _values_file = values_file;
        let values_slice: &[NumberType] = cast_slice(&values[..]);
        let height = values_slice
            .par_iter()
            .max()
            .map_or(0usize, |max| max.bit_width());

        let mut zeros_count_per_layer = Vec::with_capacity(height);
        let mut layer_blocks_vec = Vec::with_capacity(height);
        for i in 0..height {
            let current_layer_bits_file = tempfile().map_err(PyOSError::new_err)?;
            current_layer_bits_file
                .set_len(
                    (len.div_ceil(BlockType::BITS as usize) * mem::size_of::<BlockType>()) as u64,
                )
                .map_err(PyOSError::new_err)?;
            #[allow(unsafe_code)]
            let mut current_layer_bits_mmap = unsafe {
                MmapMut::map_mut(&current_layer_bits_file).map_err(PyOSError::new_err)?
            };
            let current_layer_bits_slice: &mut [BlockType] =
                cast_slice_mut(&mut current_layer_bits_mmap[..]);

            let values_slice: &[NumberType] = cast_slice(&values[..]);
            for (block_index, block) in current_layer_bits_slice.iter_mut().enumerate() {
                let start = block_index * BlockType::BITS as usize;
                let end = (start + BlockType::BITS as usize).min(len);
                for (j, &value) in values_slice[start..end].iter().enumerate() {
                    if ((value >> (height - i - 1)) & NumberType::one()).is_one() {
                        *block |= BlockType::one() << j;
                    }
                }
            }

            let next_values_file = tempfile().map_err(PyOSError::new_err)?;
            next_values_file
                .set_len((len * mem::size_of::<NumberType>()) as u64)
                .map_err(PyOSError::new_err)?;
            #[allow(unsafe_code)]
            let mut next_values_mmap =
                unsafe { MmapMut::map_mut(&next_values_file).map_err(PyOSError::new_err)? };
            let next_values_slice: &mut [NumberType] = cast_slice_mut(&mut next_values_mmap[..]);

            let zeros_count = len
                - current_layer_bits_slice
                    .par_iter()
                    .map(|block| block.count_ones())
                    .sum::<u32>() as usize;
            let mut zero_index = 0usize;
            let mut one_index = zeros_count;
            for (bit, &value) in iter::zip(
                current_layer_bits_slice
                    .iter()
                    .flat_map(|block| {
                        (0..BlockType::BITS as usize)
                            .map(move |i| ((block >> i) & BlockType::one()).is_one())
                    })
                    .take(len),
                values_slice.iter(),
            ) {
                if bit {
                    next_values_slice[one_index] = value;
                    one_index += 1;
                } else {
                    next_values_slice[zero_index] = value;
                    zero_index += 1;
                }
            }

            zeros_count_per_layer.push(zeros_count);
            layer_blocks_vec.push((
                current_layer_bits_mmap
                    .make_read_only()
                    .map_err(PyOSError::new_err)?,
                current_layer_bits_file,
            ));
            values = next_values_mmap;
            _values_file = next_values_file;
        }

        let layers = layer_blocks_vec
            .into_par_iter()
            .map(|(blocks, blocks_file)| DiskBitVector::new(blocks, blocks_file, len))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            layers,
            zeros_count_per_layer,
            height,
            len,
            phantom: marker::PhantomData,
        })
    }

    pub(crate) fn try_clone(&self) -> PyResult<Self> {
        let layers = self
            .layers
            .par_iter()
            .map(|layer| layer.try_clone())
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            layers,
            zeros_count_per_layer: self.zeros_count_per_layer.clone(),
            height: self.height,
            len: self.len,
            phantom: marker::PhantomData,
        })
    }
}

impl<NumberType> WaveletMatrixTrait<NumberType, DiskBitVector> for DiskWaveletMatrix<NumberType>
where
    NumberType:
        PrimInt + Unsigned + BitWidth + ops::BitOrAssign + ops::ShlAssign<usize> + Send + Sync,
{
    #[inline]
    fn get_layers(&self) -> &[DiskBitVector] {
        &self.layers
    }

    #[inline]
    fn get_zeros_count_per_layer(&self) -> &[usize] {
        &self.zeros_count_per_layer
    }

    #[inline]
    fn max_bit(&self) -> usize {
        self.height
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use pyo3::Python;

    use super::*;

    fn create_u32() -> DiskWaveletMatrix<u32> {
        let elements: Vec<u32> = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];

        let file = tempfile().unwrap();
        file.set_len((elements.len() * mem::size_of::<u32>()) as u64)
            .unwrap();
        #[allow(unsafe_code)]
        let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let mmap_slice: &mut [u32] = cast_slice_mut(&mut mmap[..]);
        mmap_slice.copy_from_slice(&elements);

        DiskWaveletMatrix::new(mmap, file).unwrap()
    }

    fn create_u128() -> DiskWaveletMatrix<u128> {
        let elements: Vec<u128> = vec![5, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0];

        let file = tempfile().unwrap();
        file.set_len((elements.len() * mem::size_of::<u128>()) as u64)
            .unwrap();
        #[allow(unsafe_code)]
        let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let mmap_slice: &mut [u128] = cast_slice_mut(&mut mmap[..]);
        mmap_slice.copy_from_slice(&elements);

        DiskWaveletMatrix::new(mmap, file).unwrap()
    }

    #[test]
    fn test_empty() {
        Python::initialize();

        let mmap_empty = MmapMut::map_anon(0).unwrap();
        let wv_u32 = DiskWaveletMatrix::<u32>::new(mmap_empty, tempfile().unwrap()).unwrap();
        assert_eq!(wv_u32.len(), 0);
        assert_eq!(wv_u32.max_bit(), 0);
        assert_eq!(wv_u32.values().unwrap(), Vec::<u32>::new());
        assert_eq!(
            wv_u32.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv_u32.rank(0u32, 0).unwrap(), 0);
        assert_eq!(
            wv_u32.select(0u32, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
        assert_eq!(
            wv_u32.topk(0, 0, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u32.range_list(0, 0).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );

        let mmap_empty = MmapMut::map_anon(0).unwrap();
        let wv_u128 = DiskWaveletMatrix::<u128>::new(mmap_empty, tempfile().unwrap()).unwrap();
        assert_eq!(wv_u128.len(), 0);
        assert_eq!(wv_u128.max_bit(), 0);
        assert_eq!(wv_u128.values().unwrap(), Vec::<u128>::new());
        assert_eq!(
            wv_u128.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
        assert_eq!(wv_u128.rank(0u128, 0).unwrap(), 0);
        assert_eq!(
            wv_u128.select(0u128, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );
        assert_eq!(
            wv_u128.topk(0, 0, 1).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
        assert_eq!(
            wv_u128.range_list(0, 0).unwrap_err().to_string(),
            "ValueError: start must be less than end"
        );
    }

    #[test]
    fn test_all_zero() {
        Python::initialize();

        let file = tempfile().unwrap();
        file.set_len(64 * mem::size_of::<u32>() as u64).unwrap();
        #[allow(unsafe_code)]
        let mut mmap_u32_all_zero = unsafe { MmapMut::map_mut(&file).unwrap() };
        let mmap_u32_slice: &mut [u32] = cast_slice_mut(&mut mmap_u32_all_zero[..]);
        mmap_u32_slice.fill(0);
        let wv_u32 = DiskWaveletMatrix::<u32>::new(mmap_u32_all_zero, file).unwrap();
        assert_eq!(wv_u32.len(), 64);
        assert_eq!(wv_u32.max_bit(), 0);
        assert_eq!(wv_u32.values().unwrap(), vec![0u32; 64]);
        assert_eq!(wv_u32.access(1).unwrap(), 0u32);
        assert_eq!(wv_u32.rank(0u32, 1).unwrap(), 1);
        assert_eq!(wv_u32.select(0u32, 1).unwrap(), Some(0));
        assert_eq!(wv_u32.topk(0, 64, 2).unwrap().len(), 1);
        assert_eq!(wv_u32.range_list(0, 64).unwrap().len(), 1);

        let file = tempfile().unwrap();
        file.set_len(64 * mem::size_of::<u128>() as u64).unwrap();
        #[allow(unsafe_code)]
        let mut mmap_u128_all_zero = unsafe { MmapMut::map_mut(&file).unwrap() };
        let mmap_u128_slice: &mut [u128] = cast_slice_mut(&mut mmap_u128_all_zero[..]);
        mmap_u128_slice.fill(0);
        let wv_u128 = DiskWaveletMatrix::<u128>::new(mmap_u128_all_zero, file).unwrap();
        assert_eq!(wv_u128.len(), 64);
        assert_eq!(wv_u128.max_bit(), 0);
        assert_eq!(wv_u128.values().unwrap(), vec![0u128; 64]);
        assert_eq!(wv_u128.access(1).unwrap(), 0u128);
        assert_eq!(wv_u128.rank(0u128, 1).unwrap(), 1);
        assert_eq!(wv_u128.select(0u128, 1).unwrap(), Some(0));
        assert_eq!(wv_u128.topk(0, 64, 2).unwrap().len(), 1);
        assert_eq!(wv_u128.range_list(0, 64).unwrap().len(), 1);
    }

    #[test]
    fn test_max_value() {
        Python::initialize();

        let file = tempfile().unwrap();
        file.set_len(64 * mem::size_of::<u32>() as u64).unwrap();
        #[allow(unsafe_code)]
        let mut mmap_u32_max_value = unsafe { MmapMut::map_mut(&file).unwrap() };
        let mmap_u32_slice: &mut [u32] = cast_slice_mut(&mut mmap_u32_max_value[..]);
        mmap_u32_slice.fill(u32::MAX);
        let wv_u32 = DiskWaveletMatrix::<u32>::new(mmap_u32_max_value, file).unwrap();
        assert_eq!(wv_u32.len(), 64);
        assert_eq!(wv_u32.max_bit(), 32);
        assert_eq!(wv_u32.values().unwrap(), vec![u32::MAX; 64]);
        assert_eq!(wv_u32.access(1).unwrap(), u32::MAX);
        assert_eq!(wv_u32.rank(u32::MAX, 1).unwrap(), 1);
        assert_eq!(wv_u32.select(u32::MAX, 1).unwrap(), Some(0));
        assert_eq!(wv_u32.topk(0, 64, 2).unwrap().len(), 1);
        assert_eq!(wv_u32.range_list(0, 64).unwrap().len(), 1);
    }

    #[test]
    fn test_values() {
        Python::initialize();

        let wv_u32 = create_u32();
        assert_eq!(
            wv_u32.values().unwrap(),
            vec![5u32, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128.values().unwrap(),
            vec![5u128, 4, 5, 5, 2, 1, 5, 6, 1, 3, 5, 0]
        );
    }

    #[test]
    fn test_access() {
        Python::initialize();

        let wv_u32 = create_u32();
        assert_eq!(wv_u32.access(6).unwrap(), 5u32);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.access(6).unwrap(), 5u128);
    }

    #[test]
    fn test_rank() {
        Python::initialize();

        let wv_u32 = create_u32();
        assert_eq!(wv_u32.rank(5u32, 9).unwrap(), 4usize);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.rank(5u128, 9).unwrap(), 4usize);
    }

    #[test]
    fn test_select() {
        Python::initialize();

        let wv_u32 = create_u32();
        assert_eq!(wv_u32.select(5u32, 4).unwrap(), Some(6usize));
        assert_eq!(wv_u32.select(5u32, 6).unwrap(), None);

        let wv_u128 = create_u128();
        assert_eq!(wv_u128.select(5u128, 4).unwrap(), Some(6usize));
        assert_eq!(wv_u128.select(5u128, 6).unwrap(), None);
    }

    #[test]
    fn test_topk() {
        Python::initialize();

        let wv_u32 = create_u32();
        assert_eq!(
            wv_u32.topk(1, 10, 2).unwrap(),
            vec![(5u32, 3usize), (1u32, 2usize),],
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128.topk(1, 10, 2).unwrap(),
            vec![(5u128, 3usize), (1u128, 2usize),],
        );
    }

    #[test]
    fn test_range_list() {
        Python::initialize();

        let wv_u32 = create_u32();
        assert_eq!(
            wv_u32.range_list(1, 9).unwrap(),
            vec![(1u32, 2usize), (2u32, 1usize), (4u32, 1usize), (5u32, 3usize), (6u32, 1usize),],
        );

        let wv_u128 = create_u128();
        assert_eq!(
            wv_u128.range_list(1, 9).unwrap(),
            vec![(1u128, 2usize), (2u128, 1usize), (4u128, 1usize), (5u128, 3usize), (6u128, 1usize),],
        );
    }
}
