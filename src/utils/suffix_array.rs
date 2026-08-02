// Adapted from: https://github.com/rust-lang-ja/ac-library-rs/blob/0cdbc5e2ad110b688b0239e0208e275dde94a1e2/src/string.rs
use std::{cmp, fs, iter, mem};

use bytemuck::{cast_slice, cast_slice_mut};
use memmap2::{Mmap, MmapMut};
use pyo3::{PyResult, exceptions::PyOSError};
use tempfile::tempfile;

fn suffix_array_naive(data: &Mmap) -> PyResult<(Mmap, fs::File)> {
    assert!(data.len().is_multiple_of(mem::size_of::<u32>()));
    let data_slice: &[u32] = cast_slice(&data[..]);
    let length = data_slice.len();

    let suffix_idx_file = tempfile().map_err(PyOSError::new_err)?;
    suffix_idx_file
        .set_len((length * mem::size_of::<usize>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut suffix_idx =
        unsafe { MmapMut::map_mut(&suffix_idx_file).map_err(PyOSError::new_err)? };
    let suffix_idx_slice: &mut [usize] = cast_slice_mut(&mut suffix_idx[..]);
    suffix_idx_slice.copy_from_slice(&(0..length).collect::<Vec<_>>());

    suffix_idx_slice.sort_by(|&left, &right| {
        if left == right {
            return cmp::Ordering::Equal;
        }
        let mut left = left;
        let mut right = right;
        while left < length && right < length {
            if data_slice[left] != data_slice[right] {
                return data_slice[left].cmp(&data_slice[right]);
            }
            left += 1;
            right += 1;
        }
        if left == length {
            cmp::Ordering::Less
        } else {
            cmp::Ordering::Greater
        }
    });

    Ok((
        suffix_idx
            .make_read_only()
            .map_err(PyOSError::new_err)?,
        suffix_idx_file,
    ))
}

fn suffix_array_doubling(data: &Mmap) -> PyResult<(Mmap, fs::File)> {
    assert!(data.len().is_multiple_of(mem::size_of::<u32>()));
    let data_slice: &[u32] = cast_slice(&data[..]);
    let length = data_slice.len();

    let suffix_idx_file = tempfile().map_err(PyOSError::new_err)?;
    suffix_idx_file
        .set_len((length * mem::size_of::<usize>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut suffix_idx =
        unsafe { MmapMut::map_mut(&suffix_idx_file).map_err(PyOSError::new_err)? };
    let suffix_idx_slice: &mut [usize] = cast_slice_mut(&mut suffix_idx[..]);
    suffix_idx_slice.copy_from_slice(&(0..length).collect::<Vec<_>>());

    let rank_file = tempfile().map_err(PyOSError::new_err)?;
    rank_file
        .set_len((length * mem::size_of::<u32>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut rank = unsafe { MmapMut::map_mut(&rank_file).map_err(PyOSError::new_err)? };
    let mut rank_slice: &mut [u32] = cast_slice_mut(&mut rank[..]);
    rank_slice.copy_from_slice(&data_slice[..]);

    let next_rank_file = tempfile().map_err(PyOSError::new_err)?;
    next_rank_file
        .set_len((length * mem::size_of::<u32>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut next_rank =
        unsafe { MmapMut::map_mut(&next_rank_file).map_err(PyOSError::new_err)? };

    let mut prefix_len = 1;

    while prefix_len < length {
        {
            let next_rank_slice: &mut [u32] = cast_slice_mut(&mut next_rank[..]);
            let compare_suffix = |&left: &usize, &right: &usize| {
                if rank_slice[left] != rank_slice[right] {
                    return rank_slice[left].cmp(&rank_slice[right]);
                }
                match (left + prefix_len < length, right + prefix_len < length) {
                    (false, false) => cmp::Ordering::Equal,
                    (false, true) => cmp::Ordering::Less,
                    (true, false) => cmp::Ordering::Greater,
                    (true, true) => {
                        rank_slice[left + prefix_len].cmp(&rank_slice[right + prefix_len])
                    }
                }
            };

            suffix_idx_slice.sort_by(compare_suffix);

            next_rank_slice[suffix_idx_slice[0]] = 0;
            for i in 1..length {
                let increment = compare_suffix(&suffix_idx_slice[i - 1], &suffix_idx_slice[i])
                    == cmp::Ordering::Less;
                next_rank_slice[suffix_idx_slice[i]] = if increment {
                    next_rank_slice[suffix_idx_slice[i - 1]] + 1
                } else {
                    next_rank_slice[suffix_idx_slice[i - 1]]
                };
            }
        }

        mem::swap(&mut rank, &mut next_rank);
        rank_slice = cast_slice_mut(&mut rank[..]);
        prefix_len += prefix_len;
    }

    Ok((
        suffix_idx
            .make_read_only()
            .map_err(PyOSError::new_err)?,
        suffix_idx_file,
    ))
}

fn suffix_array_induced_sorting(data: &Mmap, alphabet_max: u32) -> PyResult<(Mmap, fs::File)> {
    assert!(data.len().is_multiple_of(mem::size_of::<u32>()));
    let data_slice: &[u32] = cast_slice(&data[..]);
    let length = data_slice.len();

    let suffix_idx_file = tempfile().map_err(PyOSError::new_err)?;
    suffix_idx_file
        .set_len((length * mem::size_of::<usize>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut suffix_idx =
        unsafe { MmapMut::map_mut(&suffix_idx_file).map_err(PyOSError::new_err)? };
    let mut suffix_idx_slice: &mut [usize] = cast_slice_mut(&mut suffix_idx[..]);

    let is_s_type_file = tempfile().map_err(PyOSError::new_err)?;
    is_s_type_file
        .set_len((length * mem::size_of::<u8>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut is_s_type =
        unsafe { MmapMut::map_mut(&is_s_type_file).map_err(PyOSError::new_err)? };
    let is_s_type_slice: &mut [u8] = cast_slice_mut(&mut is_s_type[..]);
    for i in (0..length - 1).rev() {
        is_s_type_slice[i] = if data_slice[i] == data_slice[i + 1] {
            is_s_type_slice[i + 1]
        } else {
            (data_slice[i] < data_slice[i + 1]) as u8
        };
    }

    let bucket_l_start_file = tempfile().map_err(PyOSError::new_err)?;
    bucket_l_start_file
        .set_len((alphabet_max + 1) as u64 * mem::size_of::<usize>() as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut bucket_l_start =
        unsafe { MmapMut::map_mut(&bucket_l_start_file).map_err(PyOSError::new_err)? };
    let bucket_l_start_slice: &mut [usize] = cast_slice_mut(&mut bucket_l_start[..]);

    let bucket_s_start_file = tempfile().map_err(PyOSError::new_err)?;
    bucket_s_start_file
        .set_len((alphabet_max + 1) as u64 * mem::size_of::<usize>() as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut bucket_s_start =
        unsafe { MmapMut::map_mut(&bucket_s_start_file).map_err(PyOSError::new_err)? };
    let bucket_s_start_slice: &mut [usize] = cast_slice_mut(&mut bucket_s_start[..]);

    for (&is_s_type, &data) in iter::zip(is_s_type_slice.iter(), data_slice) {
        if is_s_type != 0 {
            bucket_l_start_slice[data as usize + 1] += 1;
        } else {
            bucket_s_start_slice[data as usize] += 1;
        }
    }
    for i in 0..=alphabet_max as usize {
        bucket_s_start_slice[i] += bucket_l_start_slice[i];
        if i < alphabet_max as usize {
            bucket_l_start_slice[i + 1] += bucket_s_start_slice[i];
        }
    }

    let bucket_cursor_file = tempfile().map_err(PyOSError::new_err)?;
    bucket_cursor_file
        .set_len((alphabet_max + 1) as u64 * mem::size_of::<usize>() as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut bucket_cursor =
        unsafe { MmapMut::map_mut(&bucket_cursor_file).map_err(PyOSError::new_err)? };
    let mut bucket_cursor_slice: &mut [usize] = cast_slice_mut(&mut bucket_cursor[..]);

    // suffix array's origin is +1
    let induced_sort =
        |suffix_idx: &mut [usize], bucket_cursor: &mut [usize], lms_positions: &[usize]| {
            suffix_idx.fill(0);
            bucket_cursor.copy_from_slice(&bucket_s_start_slice);
            for &lms_pos in lms_positions {
                if lms_pos == length {
                    continue;
                }
                let pos = bucket_cursor[data_slice[lms_pos] as usize];
                bucket_cursor[data_slice[lms_pos] as usize] += 1;
                suffix_idx[pos] = lms_pos + 1;
            }
            bucket_cursor.copy_from_slice(&bucket_l_start_slice);
            let pos = bucket_cursor[data_slice[length - 1] as usize];
            bucket_cursor[data_slice[length - 1] as usize] += 1;
            suffix_idx[pos] = length;
            for i in 0..length {
                let sa_value = suffix_idx[i];
                if sa_value > 1 && is_s_type_slice[sa_value - 2] == 0 {
                    let old = bucket_cursor[data_slice[sa_value - 2] as usize];
                    bucket_cursor[data_slice[sa_value - 2] as usize] += 1;
                    suffix_idx[old] = sa_value - 1;
                }
            }
            bucket_cursor.copy_from_slice(&bucket_l_start_slice);
            for i in (0..length).rev() {
                let sa_value = suffix_idx[i];
                if sa_value > 1 && is_s_type_slice[sa_value - 2] != 0 {
                    bucket_cursor[data_slice[sa_value - 2] as usize + 1] -= 1;
                    let pos = bucket_cursor[data_slice[sa_value - 2] as usize + 1];
                    suffix_idx[pos] = sa_value - 1;
                }
            }
        };

    // origin of lms_index is +1
    let lms_index_file = tempfile().map_err(PyOSError::new_err)?;
    lms_index_file
        .set_len(((length + 1) * mem::size_of::<usize>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut lms_index =
        unsafe { MmapMut::map_mut(&lms_index_file).map_err(PyOSError::new_err)? };
    let lms_index_slice: &mut [usize] = cast_slice_mut(&mut lms_index[..]);

    let mut num_lms = 0usize;
    for i in 1..length {
        if is_s_type_slice[i - 1] == 0 && is_s_type_slice[i] != 0 {
            lms_index_slice[i] = num_lms + 1;
            num_lms += 1;
        }
    }

    let lms_positions_file = tempfile().map_err(PyOSError::new_err)?;
    lms_positions_file
        .set_len((num_lms * mem::size_of::<usize>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut lms_positions =
        unsafe { MmapMut::map_mut(&lms_positions_file).map_err(PyOSError::new_err)? };
    let lms_positions_slice: &mut [usize] = cast_slice_mut(&mut lms_positions[..]);
    lms_positions_slice.copy_from_slice(
        &(1..length)
            .filter(|&i| is_s_type_slice[i - 1] == 0 && is_s_type_slice[i] != 0)
            .collect::<Vec<usize>>()
    );

    induced_sort(
        &mut suffix_idx_slice,
        &mut bucket_cursor_slice,
        &lms_positions_slice,
    );

    if num_lms > 0 {
        let sorted_lms_positions_file = tempfile().map_err(PyOSError::new_err)?;
        sorted_lms_positions_file
            .set_len((num_lms * mem::size_of::<usize>()) as u64)
            .map_err(PyOSError::new_err)?;
        #[allow(unsafe_code)]
        let mut sorted_lms_positions = unsafe {
            MmapMut::map_mut(&sorted_lms_positions_file).map_err(PyOSError::new_err)?
        };
        let sorted_lms_positions_slice: &mut [usize] = cast_slice_mut(&mut sorted_lms_positions[..]);
        sorted_lms_positions_slice.copy_from_slice(
            &suffix_idx_slice
                .iter()
                .filter(|&&sa_value| lms_index_slice[sa_value - 1] > 0)
                .map(|&sa_value| sa_value - 1)
                .collect::<Vec<_>>()
        );

        let reduced_data_file = tempfile().map_err(PyOSError::new_err)?;
        reduced_data_file
            .set_len((num_lms * mem::size_of::<u32>()) as u64)
            .map_err(PyOSError::new_err)?;
        #[allow(unsafe_code)]
        let mut reduced_data =
            unsafe { MmapMut::map_mut(&reduced_data_file).map_err(PyOSError::new_err)? };
        let reduced_data_slice: &mut [u32] = cast_slice_mut(&mut reduced_data[..]);

        let mut reduced_alphabet_max = 0;
        reduced_data_slice[lms_index_slice[sorted_lms_positions_slice[0]] - 1] = 0;

        for i in 1..num_lms {
            let mut prev_pos = sorted_lms_positions_slice[i - 1];
            let mut curr_pos = sorted_lms_positions_slice[i];
            let prev_end = if lms_index_slice[prev_pos] < num_lms {
                lms_positions_slice[lms_index_slice[prev_pos]]
            } else {
                length
            };
            let curr_end = if lms_index_slice[curr_pos] < num_lms {
                lms_positions_slice[lms_index_slice[curr_pos]]
            } else {
                length
            };

            let is_same_lms_substring = if prev_end - prev_pos != curr_end - curr_pos {
                false
            } else {
                while prev_pos < prev_end && data_slice[prev_pos] == data_slice[curr_pos] {
                    prev_pos += 1;
                    curr_pos += 1;
                }
                prev_pos != length && data_slice[prev_pos] == data_slice[curr_pos]
            };

            if !is_same_lms_substring {
                reduced_alphabet_max += 1;
            }
            reduced_data_slice[lms_index_slice[sorted_lms_positions_slice[i]] - 1] =
                reduced_alphabet_max;
        }

        let (reduced_suffix_array, _) = suffix_array_inner(
            &reduced_data
                .make_read_only()
                .map_err(PyOSError::new_err)?,
            reduced_alphabet_max,
        )?;
        let reduced_suffix_array_slice: &[usize] = cast_slice(&reduced_suffix_array[..]);
        for (i, &reduced_sa_value) in reduced_suffix_array_slice.iter().enumerate() {
            sorted_lms_positions_slice[i] = lms_positions_slice[reduced_sa_value];
        }

        induced_sort(
            &mut suffix_idx_slice,
            &mut bucket_cursor_slice,
            &sorted_lms_positions_slice,
        );
    }
    for x in &mut suffix_idx_slice[..] {
        *x -= 1;
    }
    Ok((
        suffix_idx
            .make_read_only()
            .map_err(PyOSError::new_err)?,
        suffix_idx_file,
    ))
}

fn suffix_array_inner(data: &Mmap, alphabet_max: u32) -> PyResult<(Mmap, fs::File)> {
    let length = data.len() / mem::size_of::<u32>();

    match length {
        0..3 => {
            let suffix_idx_file = tempfile().map_err(PyOSError::new_err)?;
            suffix_idx_file
                .set_len((length * mem::size_of::<usize>()) as u64)
                .map_err(PyOSError::new_err)?;
            #[allow(unsafe_code)]
            let mut suffix_idx =
                unsafe { MmapMut::map_mut(&suffix_idx_file).map_err(PyOSError::new_err)? };
            let suffix_idx_data: &mut [usize] = cast_slice_mut(&mut suffix_idx[..]);
            match length {
                0 => Ok((
                    suffix_idx
                        .make_read_only()
                        .map_err(PyOSError::new_err)?,
                    suffix_idx_file,
                )),
                1 => {
                    suffix_idx_data[0] = 0;
                    Ok((
                        suffix_idx
                            .make_read_only()
                            .map_err(PyOSError::new_err)?,
                        suffix_idx_file,
                    ))
                }
                2 => {
                    let data: &[u32] = cast_slice(&data[..]);
                    if data[0] < data[1] {
                        suffix_idx_data.copy_from_slice(&[0, 1]);
                    } else {
                        suffix_idx_data.copy_from_slice(&[1, 0]);
                    }
                    Ok((
                        suffix_idx
                            .make_read_only()
                            .map_err(PyOSError::new_err)?,
                        suffix_idx_file,
                    ))
                }
                _ => unreachable!(),
            }
        }
        3..10 => suffix_array_naive(data),
        10..40 => suffix_array_doubling(data),
        _ => suffix_array_induced_sorting(data, alphabet_max),
    }
}

pub(crate) fn suffix_array_vec(data: &Vec<u32>) -> PyResult<Vec<usize>> {
    let data_file = tempfile().map_err(PyOSError::new_err)?;
    data_file
        .set_len((data.len() * mem::size_of::<u32>()) as u64)
        .map_err(PyOSError::new_err)?;
    #[allow(unsafe_code)]
    let mut data_mmap = unsafe { MmapMut::map_mut(&data_file).map_err(PyOSError::new_err)? };
    let data_mmap_data: &mut [u32] = cast_slice_mut(&mut data_mmap[..]);
    data_mmap_data.copy_from_slice(data.as_slice());

    let &alphabet_max = data.iter().max().unwrap_or(&0);

    let (result_mmap, _) = suffix_array_inner(
        &data_mmap
            .make_read_only()
            .map_err(PyOSError::new_err)?,
        alphabet_max,
    )?;
    Ok(cast_slice::<u8, usize>(&result_mmap[..])
        .iter()
        .copied()
        .collect::<Vec<usize>>())
}

pub(crate) fn suffix_array_mmap(data: &Mmap) -> PyResult<(Mmap, fs::File)> {
    let &alphabet_max = cast_slice::<u8, u32>(&data[..]).iter().max().unwrap_or(&0);
    suffix_array_inner(data, alphabet_max)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    fn verify_all(array: &[u32], expected_idx: &[usize]) {
        let &alphabet_max = array.iter().max().unwrap_or(&0);

        let create_array_mmap = |array: &[u32]| -> Mmap {
            let mut array_mmap = MmapMut::map_anon(array.len() * mem::size_of::<u32>()).unwrap();
            let array_data: &mut [u32] = cast_slice_mut(&mut array_mmap);
            array_data.copy_from_slice(&array);
            array_mmap.make_read_only().unwrap()
        };

        let mut expeted_idx_mmap =
            MmapMut::map_anon(expected_idx.len() * mem::size_of::<usize>()).unwrap();
        let expeted_idx_data: &mut [usize] = cast_slice_mut(&mut expeted_idx_mmap);
        expeted_idx_data.copy_from_slice(&expected_idx);
        let expected_idx_mmap = expeted_idx_mmap.make_read_only().unwrap();

        let (suffix_idx_doubling, _) = suffix_array_doubling(&create_array_mmap(array)).unwrap();
        assert_eq!(suffix_idx_doubling.to_vec(), expected_idx_mmap.to_vec());
        let (suffix_idx_naive, _) = suffix_array_naive(&create_array_mmap(array)).unwrap();
        assert_eq!(suffix_idx_naive.to_vec(), expected_idx_mmap.to_vec());
        let (suffix_idx_induced_sorting, _) =
            suffix_array_induced_sorting(&create_array_mmap(array), alphabet_max).unwrap();
        assert_eq!(
            suffix_idx_induced_sorting.to_vec(),
            expected_idx_mmap.to_vec()
        );
        let (suffix_idx, _) = suffix_array_inner(&create_array_mmap(array), alphabet_max).unwrap();
        assert_eq!(suffix_idx.to_vec(), expected_idx_mmap.to_vec());
        let suffix_idx_vec = suffix_array_vec(&array.to_vec()).unwrap();
        assert_eq!(suffix_idx_vec, expected_idx);
        let (suffix_idx_mmap, _) = suffix_array_mmap(&create_array_mmap(array)).unwrap();
        assert_eq!(suffix_idx_mmap.to_vec(), expected_idx_mmap.to_vec());
    }

    #[test]
    fn test_suffix_array_0() {
        let array = [0, 1, 2, 3, 4, 5];

        verify_all(&array, &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_suffix_array_1() {
        let str = "abracadabra";
        let array = str.bytes().map(|byte| byte as u32).collect::<Vec<_>>();

        verify_all(&array, &[10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2]);
    }

    #[test]
    fn test_suffix_array_2() {
        let str = "mmiissiissiippii"; // an example taken from https://mametter.hatenablog.com/entry/20180130/p1
        let array = str.bytes().map(|byte| byte as u32).collect::<Vec<_>>();

        verify_all(
            &array,
            &[15, 14, 10, 6, 2, 11, 7, 3, 1, 0, 13, 12, 9, 5, 8, 4],
        );
    }

    #[test]
    fn test_suffix_array_3() {
        let str = "mississippi".repeat(50);
        let array = str.bytes().map(|byte| byte as u32).collect::<Vec<_>>();

        verify_all(
            &array,
            &[
                549, 538, 527, 516, 505, 494, 483, 472, 461, 450, 439, 428, 417, 406, 395, 384,
                373, 362, 351, 340, 329, 318, 307, 296, 285, 274, 263, 252, 241, 230, 219, 208,
                197, 186, 175, 164, 153, 142, 131, 120, 109, 98, 87, 76, 65, 54, 43, 32, 21, 10,
                546, 535, 524, 513, 502, 491, 480, 469, 458, 447, 436, 425, 414, 403, 392, 381,
                370, 359, 348, 337, 326, 315, 304, 293, 282, 271, 260, 249, 238, 227, 216, 205,
                194, 183, 172, 161, 150, 139, 128, 117, 106, 95, 84, 73, 62, 51, 40, 29, 18, 7,
                543, 532, 521, 510, 499, 488, 477, 466, 455, 444, 433, 422, 411, 400, 389, 378,
                367, 356, 345, 334, 323, 312, 301, 290, 279, 268, 257, 246, 235, 224, 213, 202,
                191, 180, 169, 158, 147, 136, 125, 114, 103, 92, 81, 70, 59, 48, 37, 26, 15, 4,
                540, 529, 518, 507, 496, 485, 474, 463, 452, 441, 430, 419, 408, 397, 386, 375,
                364, 353, 342, 331, 320, 309, 298, 287, 276, 265, 254, 243, 232, 221, 210, 199,
                188, 177, 166, 155, 144, 133, 122, 111, 100, 89, 78, 67, 56, 45, 34, 23, 12, 1,
                539, 528, 517, 506, 495, 484, 473, 462, 451, 440, 429, 418, 407, 396, 385, 374,
                363, 352, 341, 330, 319, 308, 297, 286, 275, 264, 253, 242, 231, 220, 209, 198,
                187, 176, 165, 154, 143, 132, 121, 110, 99, 88, 77, 66, 55, 44, 33, 22, 11, 0, 548,
                537, 526, 515, 504, 493, 482, 471, 460, 449, 438, 427, 416, 405, 394, 383, 372,
                361, 350, 339, 328, 317, 306, 295, 284, 273, 262, 251, 240, 229, 218, 207, 196,
                185, 174, 163, 152, 141, 130, 119, 108, 97, 86, 75, 64, 53, 42, 31, 20, 9, 547,
                536, 525, 514, 503, 492, 481, 470, 459, 448, 437, 426, 415, 404, 393, 382, 371,
                360, 349, 338, 327, 316, 305, 294, 283, 272, 261, 250, 239, 228, 217, 206, 195,
                184, 173, 162, 151, 140, 129, 118, 107, 96, 85, 74, 63, 52, 41, 30, 19, 8, 545,
                534, 523, 512, 501, 490, 479, 468, 457, 446, 435, 424, 413, 402, 391, 380, 369,
                358, 347, 336, 325, 314, 303, 292, 281, 270, 259, 248, 237, 226, 215, 204, 193,
                182, 171, 160, 149, 138, 127, 116, 105, 94, 83, 72, 61, 50, 39, 28, 17, 6, 542,
                531, 520, 509, 498, 487, 476, 465, 454, 443, 432, 421, 410, 399, 388, 377, 366,
                355, 344, 333, 322, 311, 300, 289, 278, 267, 256, 245, 234, 223, 212, 201, 190,
                179, 168, 157, 146, 135, 124, 113, 102, 91, 80, 69, 58, 47, 36, 25, 14, 3, 544,
                533, 522, 511, 500, 489, 478, 467, 456, 445, 434, 423, 412, 401, 390, 379, 368,
                357, 346, 335, 324, 313, 302, 291, 280, 269, 258, 247, 236, 225, 214, 203, 192,
                181, 170, 159, 148, 137, 126, 115, 104, 93, 82, 71, 60, 49, 38, 27, 16, 5, 541,
                530, 519, 508, 497, 486, 475, 464, 453, 442, 431, 420, 409, 398, 387, 376, 365,
                354, 343, 332, 321, 310, 299, 288, 277, 266, 255, 244, 233, 222, 211, 200, 189,
                178, 167, 156, 145, 134, 123, 112, 101, 90, 79, 68, 57, 46, 35, 24, 13, 2,
            ],
        );
    }

    #[test]
    fn test_suffix_array_4() {
        let str_list = ["banana", "ananas", "abracadabra", "mississippi"];
        let array = str_list
            .into_iter()
            .flat_map(|str| str.chars().map(|c| c as u32).chain(iter::once(0)))
            .collect::<Vec<_>>();

        verify_all(
            &array,
            &[
                37, 13, 6, 25, 5, 24, 21, 14, 17, 19, 3, 1, 7, 9, 11, 0, 22, 15, 18, 20, 36, 33,
                30, 27, 26, 4, 2, 8, 10, 35, 34, 23, 16, 12, 32, 29, 31, 28,
            ],
        );
    }
}
