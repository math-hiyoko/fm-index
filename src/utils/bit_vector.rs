use std::iter;

use num_integer::Integer;
use num_traits::{One, Zero};
use pyo3::{
    PyResult,
    exceptions::{PyIndexError, PyValueError},
};
use serde::{Deserialize, Serialize};

use crate::utils::bit_select::BitSelect;

pub(super) type BlockType = u128;
const SELECT_INDEX_INTERBVAL: usize = 1 << 14;

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct BitVector {
    len: usize,
    blocks: Vec<BlockType>,
    ranks: Vec<usize>,
    select_index: [Vec<usize>; 2],
}

impl BitVector {
    pub(super) fn new(blocks: Vec<BlockType>, len: usize) -> PyResult<Self> {
        // Build the rank index structure.
        let ranks: Vec<usize> = iter::once(0usize)
            .chain(blocks.iter().scan(0usize, |acc, block| {
                *acc += block.count_ones() as usize;
                Some(*acc)
            }))
            .collect();

        let mut select_index = [
            Vec::with_capacity((len - ranks.last().unwrap()) / SELECT_INDEX_INTERBVAL + 1),
            Vec::with_capacity((ranks.last().unwrap() / SELECT_INDEX_INTERBVAL) + 1),
        ];
        for select_index_inner in select_index.iter_mut() {
            select_index_inner.push(0);
        }
        let mut count = [0usize, 0usize];
        for index in 0..len {
            let (block_index, bit_index) = index.div_rem(&(BlockType::BITS as usize));
            let bit = ((blocks[block_index] >> bit_index) & BlockType::one()).is_one() as usize;
            count[bit] += 1;
            if count[bit].is_multiple_of(SELECT_INDEX_INTERBVAL) {
                select_index[bit].push(index);
            }
        }
        for select_index_inner in select_index.iter_mut() {
            select_index_inner.push(len);
        }

        Ok(Self {
            len,
            blocks,
            ranks,
            select_index,
        })
    }

    #[inline]
    pub(super) fn access(&self, index: usize) -> PyResult<bool> {
        if index >= self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        let (block_index, bit_index) = index.div_rem(&(BlockType::BITS as usize));
        Ok(((self.blocks[block_index] >> bit_index) & BlockType::one()).is_one())
    }

    #[inline]
    pub(super) fn values(&self) -> PyResult<Vec<BlockType>> {
        Ok(self.blocks.clone())
    }

    #[inline]
    pub(super) fn rank(&self, bit: bool, end: usize) -> PyResult<usize> {
        if end > self.len {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        if self.len.is_zero() {
            return Ok(0);
        }
        if !bit {
            return Ok(end - self.rank(true, end)?);
        }

        let (block_index, bit_index) = end.div_rem(&(BlockType::BITS as usize));
        let mut rank = self.ranks[block_index];
        if block_index < self.blocks.len() {
            rank += (self.blocks[block_index]
                & ((BlockType::one() << bit_index) - BlockType::one()))
            .count_ones() as usize;
        }
        Ok(rank)
    }

    #[inline]
    pub(super) fn select(&self, bit: bool, mut kth: usize) -> PyResult<Option<usize>> {
        if kth.is_zero() {
            return Err(PyValueError::new_err("kth must be greater than 0"));
        }
        if kth > self.rank(bit, self.len)? {
            return Ok(None);
        }

        let block_index = {
            let mut left = self.select_index[bit as usize][(kth - 1) / SELECT_INDEX_INTERBVAL]
                / (BlockType::BITS as usize);
            let mut right = self.select_index[bit as usize][kth / SELECT_INDEX_INTERBVAL + 1]
                .div_ceil(BlockType::BITS as usize);
            debug_assert!(right <= self.blocks.len());
            while left + 1 < right {
                let mid = (left + right) / 2;
                let rank_at_mid = if bit {
                    self.ranks[mid]
                } else {
                    mid * (BlockType::BITS as usize) - self.ranks[mid]
                };
                if rank_at_mid < kth {
                    left = mid;
                } else {
                    right = mid;
                }
            }
            left
        };

        kth -= if bit {
            self.ranks[block_index]
        } else {
            block_index * (BlockType::BITS as usize) - self.ranks[block_index]
        };
        let index = self.blocks[block_index].bit_select(bit, kth).unwrap()
            + block_index * (BlockType::BITS as usize);

        Ok(Some(index))
    }
}

#[cfg(test)]
mod tests {
    use pyo3::Python;

    use super::*;

    fn create_test_bit_vector() -> BitVector {
        let bits = [true, false, true, true, false, true, false, false].repeat(999);
        let blocks = bits
            .chunks(BlockType::BITS as usize)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(BlockType::zero(), |acc, (i, &bit)| {
                        acc | ((bit as BlockType) << i)
                    })
            })
            .collect::<Vec<_>>();
        BitVector::new(blocks, bits.len()).unwrap()
    }

    #[test]
    fn test_empty_bit_vector() {
        Python::initialize();

        let bv = BitVector::new(vec![], 0).unwrap();

        // Access should fail on empty vector
        assert_eq!(
            bv.access(0).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );

        // Values should return empty vector
        assert_eq!(bv.values().unwrap(), Vec::<BlockType>::new());

        // Rank should return 0
        assert_eq!(bv.rank(true, 0).unwrap(), 0);
        assert_eq!(bv.rank(false, 0).unwrap(), 0);

        // Select should return None
        assert_eq!(bv.select(true, 1).unwrap(), None);
        assert_eq!(bv.select(false, 1).unwrap(), None);
    }

    #[test]
    fn test_exact_block_size() {
        Python::initialize();

        let bits = vec![true; 1024];
        let blocks = bits
            .chunks(BlockType::BITS as usize)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(BlockType::zero(), |acc, (i, &bit)| {
                        acc | ((bit as BlockType) << i)
                    })
            })
            .collect::<Vec<_>>();
        let bv = BitVector::new(blocks.clone(), bits.len()).unwrap();

        assert_eq!(bv.values().unwrap(), blocks);
        for i in 0..1024 {
            assert!(bv.access(i).unwrap());
            assert_eq!(bv.rank(true, i + 1).unwrap(), i + 1);
            assert_eq!(bv.rank(false, i + 1).unwrap(), 0);
            assert_eq!(bv.select(true, i + 1).unwrap(), Some(i));
            assert_eq!(bv.select(false, i + 1).unwrap(), None);
        }
    }

    #[test]
    fn test_access_operation() {
        Python::initialize();

        let bv = create_test_bit_vector();

        assert!(bv.access(0).unwrap());
        assert!(!bv.access(1001).unwrap());
        assert!(bv.access(2002).unwrap());
        assert!(bv.access(3003).unwrap());
        assert!(!bv.access(4004).unwrap());
        assert!(bv.access(5005).unwrap());
        assert!(!bv.access(6006).unwrap());
        assert!(!bv.access(7007).unwrap());

        // Out of bounds
        assert_eq!(
            bv.access(7992).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
    }

    #[test]
    fn test_values_retrieval() {
        Python::initialize();

        let bv = create_test_bit_vector();

        let expected = [true, false, true, true, false, true, false, false].repeat(999);
        let expected_blocks = expected
            .chunks(BlockType::BITS as usize)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(BlockType::zero(), |acc, (i, &bit)| {
                        acc | ((bit as BlockType) << i)
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(bv.values().unwrap(), expected_blocks);
    }

    #[test]
    fn test_rank_operation() {
        Python::initialize();

        let bv = create_test_bit_vector();

        // Rank for true bits
        assert_eq!(bv.rank(true, 0).unwrap(), 0);
        assert_eq!(bv.rank(true, 1001).unwrap(), 501);
        assert_eq!(bv.rank(true, 2002).unwrap(), 1001);
        assert_eq!(bv.rank(true, 3003).unwrap(), 1502);
        assert_eq!(bv.rank(true, 4004).unwrap(), 2003);
        assert_eq!(bv.rank(true, 5005).unwrap(), 2503);
        assert_eq!(bv.rank(true, 6006).unwrap(), 3004);
        assert_eq!(bv.rank(true, 7007).unwrap(), 3504);
        assert_eq!(bv.rank(true, 7992).unwrap(), 3996);

        // Out of bounds for true
        assert_eq!(
            bv.rank(true, 7993).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );

        // Rank for false bits
        assert_eq!(bv.rank(false, 0).unwrap(), 0);
        assert_eq!(bv.rank(false, 1001).unwrap(), 500);
        assert_eq!(bv.rank(false, 2002).unwrap(), 1001);
        assert_eq!(bv.rank(false, 3003).unwrap(), 1501);
        assert_eq!(bv.rank(false, 4004).unwrap(), 2001);
        assert_eq!(bv.rank(false, 5005).unwrap(), 2502);
        assert_eq!(bv.rank(false, 6006).unwrap(), 3002);
        assert_eq!(bv.rank(false, 7007).unwrap(), 3503);
        assert_eq!(bv.rank(false, 7992).unwrap(), 3996);

        // Out of bounds for false
        assert_eq!(
            bv.rank(false, 7993).unwrap_err().to_string(),
            "IndexError: index out of bounds"
        );
    }

    #[test]
    fn test_select_operation() {
        Python::initialize();

        let bv = create_test_bit_vector();

        // Invalid kth=0 for true
        assert_eq!(
            bv.select(true, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );

        // Valid selections for true bits
        assert_eq!(bv.select(true, 1).unwrap(), Some(0));
        assert_eq!(bv.select(true, 1000).unwrap(), Some(1997));
        assert_eq!(bv.select(true, 2000).unwrap(), Some(3997));
        assert_eq!(bv.select(true, 3000).unwrap(), Some(5997));
        assert_eq!(bv.select(true, 3996).unwrap(), Some(7989));

        // Out of range for true
        assert_eq!(bv.select(true, 3997).unwrap(), None);

        // Invalid kth=0 for false
        assert_eq!(
            bv.select(false, 0).unwrap_err().to_string(),
            "ValueError: kth must be greater than 0"
        );

        // Valid selections for false bits
        assert_eq!(bv.select(false, 1).unwrap(), Some(1));
        assert_eq!(bv.select(false, 1000).unwrap(), Some(1999));
        assert_eq!(bv.select(false, 2000).unwrap(), Some(3999));
        assert_eq!(bv.select(false, 3000).unwrap(), Some(5999));
        assert_eq!(bv.select(false, 3996).unwrap(), Some(7991));

        // Out of range for false
        assert_eq!(bv.select(false, 3997).unwrap(), None);
    }
}
