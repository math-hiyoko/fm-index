// Adapted from: https://github.com/rust-lang-ja/ac-library-rs/blob/0cdbc5e2ad110b688b0239e0208e275dde94a1e2/src/string.rs
use std::{cmp, fmt, iter, mem, ops};

use num_traits::PrimInt;
use rayon::prelude::*;

fn suffix_array_naive<IndexType>(data: Vec<IndexType>) -> Vec<IndexType>
where
    usize: TryFrom<IndexType>,
    IndexType: PrimInt + ops::AddAssign + TryFrom<usize>,
    <usize as TryFrom<IndexType>>::Error: fmt::Debug,
    <IndexType as TryFrom<usize>>::Error: fmt::Debug,
{
    let length = data.len().try_into().unwrap();

    let mut suffix_idx = (0..data.len())
        .map(|value| value.try_into().unwrap())
        .collect::<Vec<_>>();
    suffix_idx.sort_by(|&(mut left): &IndexType, &(mut right): &IndexType| {
        if left == right {
            return cmp::Ordering::Equal;
        }
        while left < length && right < length {
            if data[usize::try_from(left).unwrap()] != data[usize::try_from(right).unwrap()] {
                return data[usize::try_from(left).unwrap()]
                    .cmp(&data[usize::try_from(right).unwrap()]);
            }
            left += IndexType::one();
            right += IndexType::one();
        }
        if left == length {
            cmp::Ordering::Less
        } else {
            cmp::Ordering::Greater
        }
    });

    suffix_idx
}

fn suffix_array_doubling<IndexType>(data: Vec<IndexType>) -> Vec<IndexType>
where
    usize: TryFrom<IndexType>,
    IndexType: PrimInt + ops::AddAssign + TryFrom<usize>,
    <usize as TryFrom<IndexType>>::Error: fmt::Debug,
    <IndexType as TryFrom<usize>>::Error: fmt::Debug,
{
    let length = data.len().try_into().unwrap();

    let mut suffix_idx = (0..data.len())
        .map(|value| value.try_into().unwrap())
        .collect::<Vec<_>>();
    let mut rank = data;

    let mut next_rank = vec![IndexType::zero(); rank.len()];
    let mut prefix_len = IndexType::one();

    while prefix_len < length {
        let compare_suffix = |&left: &IndexType, &right: &IndexType| {
            if rank[usize::try_from(left).unwrap()] != rank[usize::try_from(right).unwrap()] {
                return rank[usize::try_from(left).unwrap()]
                    .cmp(&rank[usize::try_from(right).unwrap()]);
            }
            match (left + prefix_len < length, right + prefix_len < length) {
                (false, false) => cmp::Ordering::Equal,
                (false, true) => cmp::Ordering::Less,
                (true, false) => cmp::Ordering::Greater,
                (true, true) => rank[usize::try_from(left + prefix_len).unwrap()]
                    .cmp(&rank[usize::try_from(right + prefix_len).unwrap()]),
            }
        };

        suffix_idx.sort_by(compare_suffix);

        next_rank[usize::try_from(suffix_idx[0]).unwrap()] = IndexType::zero();
        for i in 1..length.try_into().unwrap() {
            let increment =
                compare_suffix(&suffix_idx[i - 1], &suffix_idx[i]) == cmp::Ordering::Less;
            next_rank[usize::try_from(suffix_idx[i]).unwrap()] = if increment {
                next_rank[usize::try_from(suffix_idx[i - 1]).unwrap()] + IndexType::one()
            } else {
                next_rank[usize::try_from(suffix_idx[i - 1]).unwrap()]
            };
        }

        mem::swap(&mut rank, &mut next_rank);
        prefix_len += prefix_len;
    }

    suffix_idx
}

fn suffix_array_induced_sorting<IndexType>(
    data: Vec<IndexType>,
    alphabet_max: IndexType,
) -> Vec<IndexType>
where
    usize: TryFrom<IndexType>,
    IndexType: PrimInt + ops::AddAssign + ops::SubAssign + TryFrom<usize>,
    <usize as TryFrom<IndexType>>::Error: fmt::Debug,
    <IndexType as TryFrom<usize>>::Error: fmt::Debug,
{
    let length: IndexType = data.len().try_into().unwrap();
    let mut suffix_idx = vec![IndexType::zero(); length.try_into().unwrap()];
    let mut is_s_type = vec![false; length.try_into().unwrap()];
    for i in (0..data.len() - 1).rev() {
        is_s_type[i] = if data[i] == data[i + 1] {
            is_s_type[i + 1]
        } else {
            data[i] < data[i + 1]
        };
    }

    let mut bucket_l_start = vec![IndexType::zero(); usize::try_from(alphabet_max).unwrap() + 1];
    let mut bucket_s_start = vec![IndexType::zero(); usize::try_from(alphabet_max).unwrap() + 1];
    for (&is_s_type, &data) in iter::zip(&is_s_type, &data) {
        if is_s_type {
            bucket_l_start[usize::try_from(data).unwrap() + 1] += IndexType::one();
        } else {
            bucket_s_start[usize::try_from(data).unwrap()] += IndexType::one();
        }
    }
    for i in 0..=usize::try_from(alphabet_max).unwrap() {
        bucket_s_start[i] += bucket_l_start[i];
        if i < usize::try_from(alphabet_max).unwrap() {
            bucket_l_start[i + 1] += bucket_s_start[i];
        }
    }

    let mut bucket_cursor = vec![IndexType::zero(); usize::try_from(alphabet_max).unwrap() + 1];
    // suffix array's origin is +1
    let induced_sort = |suffix_idx: &mut [IndexType],
                        bucket_cursor: &mut [IndexType],
                        lms_positions: &[IndexType]| {
        suffix_idx.fill(IndexType::zero());
        bucket_cursor.copy_from_slice(&bucket_s_start);
        for &lms_pos in lms_positions {
            if lms_pos == length {
                continue;
            }
            let pos =
                bucket_cursor[usize::try_from(data[usize::try_from(lms_pos).unwrap()]).unwrap()];
            bucket_cursor[usize::try_from(data[usize::try_from(lms_pos).unwrap()]).unwrap()] +=
                IndexType::one();
            suffix_idx[usize::try_from(pos).unwrap()] = lms_pos + IndexType::one();
        }
        bucket_cursor.copy_from_slice(&bucket_l_start);
        let pos =
            bucket_cursor[usize::try_from(data[usize::try_from(length).unwrap() - 1]).unwrap()];
        bucket_cursor[usize::try_from(data[usize::try_from(length).unwrap() - 1]).unwrap()] +=
            IndexType::one();
        suffix_idx[usize::try_from(pos).unwrap()] = length;
        for i in 0..usize::try_from(length).unwrap() {
            let sa_value = suffix_idx[i];
            if sa_value > IndexType::one() && !is_s_type[usize::try_from(sa_value).unwrap() - 2] {
                let old = bucket_cursor
                    [usize::try_from(data[usize::try_from(sa_value).unwrap() - 2]).unwrap()];
                bucket_cursor
                    [usize::try_from(data[usize::try_from(sa_value).unwrap() - 2]).unwrap()] +=
                    IndexType::one();
                suffix_idx[usize::try_from(old).unwrap()] = sa_value - IndexType::one();
            }
        }
        bucket_cursor.copy_from_slice(&bucket_l_start);
        for i in (0..usize::try_from(length).unwrap()).rev() {
            let sa_value = suffix_idx[i];
            if sa_value > IndexType::one() && is_s_type[usize::try_from(sa_value).unwrap() - 2] {
                bucket_cursor
                    [usize::try_from(data[usize::try_from(sa_value).unwrap() - 2]).unwrap() + 1] -=
                    IndexType::one();
                let pos: IndexType = bucket_cursor
                    [usize::try_from(data[usize::try_from(sa_value).unwrap() - 2]).unwrap() + 1];
                suffix_idx[usize::try_from(pos).unwrap()] = sa_value - IndexType::one();
            }
        }
    };

    // origin of lms_index is +1
    let mut lms_index = vec![IndexType::zero(); usize::try_from(length).unwrap() + 1];
    let mut num_lms = IndexType::zero();
    for i in 1..usize::try_from(length).unwrap() {
        if !is_s_type[i - 1] && is_s_type[i] {
            lms_index[i] = num_lms + IndexType::one();
            num_lms += IndexType::one();
        }
    }
    let lms_positions = (1..usize::try_from(length).unwrap())
        .filter(|&i| !is_s_type[i - 1] && is_s_type[i])
        .map(|i| i.try_into().unwrap())
        .collect::<Vec<_>>();
    induced_sort(&mut suffix_idx, &mut bucket_cursor, &lms_positions);

    if num_lms > IndexType::zero() {
        let mut sorted_lms_positions = suffix_idx
            .iter()
            .filter_map(|&sa_value| {
                if lms_index[usize::try_from(sa_value).unwrap() - 1] > IndexType::zero() {
                    Some(sa_value - IndexType::one())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut reduced_data = vec![IndexType::zero(); usize::try_from(num_lms).unwrap()];
        let mut reduced_alphabet_max = IndexType::zero();
        reduced_data[usize::try_from(
            lms_index[usize::try_from(sorted_lms_positions[0]).unwrap()],
        )
        .unwrap()
            - 1] = IndexType::zero();
        for i in 1..usize::try_from(num_lms).unwrap() {
            let mut prev_pos = sorted_lms_positions[i - 1];
            let mut curr_pos = sorted_lms_positions[i];
            let prev_end = if lms_index[usize::try_from(prev_pos).unwrap()] < num_lms {
                lms_positions
                    [usize::try_from(lms_index[usize::try_from(prev_pos).unwrap()]).unwrap()]
            } else {
                length
            };
            let curr_end = if lms_index[usize::try_from(curr_pos).unwrap()] < num_lms {
                lms_positions
                    [usize::try_from(lms_index[usize::try_from(curr_pos).unwrap()]).unwrap()]
            } else {
                length
            };
            let is_same_lms_substring = if prev_end - prev_pos != curr_end - curr_pos {
                false
            } else {
                while prev_pos < prev_end
                    && data[usize::try_from(prev_pos).unwrap()]
                        == data[usize::try_from(curr_pos).unwrap()]
                {
                    prev_pos += IndexType::one();
                    curr_pos += IndexType::one();
                }
                prev_pos != length
                    && data[usize::try_from(prev_pos).unwrap()]
                        == data[usize::try_from(curr_pos).unwrap()]
            };

            if !is_same_lms_substring {
                reduced_alphabet_max += IndexType::one();
            }
            reduced_data[usize::try_from(
                lms_index[usize::try_from(sorted_lms_positions[i]).unwrap()],
            )
            .unwrap()
                - 1] = reduced_alphabet_max;
        }

        drop(lms_index);
        let reduced_suffix_array = suffix_array_inner(reduced_data, reduced_alphabet_max);
        for (i, reduced_sa_value) in reduced_suffix_array.into_iter().enumerate() {
            sorted_lms_positions[i] = lms_positions[usize::try_from(reduced_sa_value).unwrap()];
        }
        drop(lms_positions);
        induced_sort(&mut suffix_idx, &mut bucket_cursor, &sorted_lms_positions);
    }
    suffix_idx.iter_mut().for_each(|x| *x -= IndexType::one());
    suffix_idx
}

fn suffix_array_inner<IndexType>(data: Vec<IndexType>, alphabet_max: IndexType) -> Vec<IndexType>
where
    usize: TryFrom<IndexType>,
    IndexType: PrimInt + ops::AddAssign + ops::SubAssign + TryFrom<usize>,
    <usize as TryFrom<IndexType>>::Error: fmt::Debug,
    <IndexType as TryFrom<usize>>::Error: fmt::Debug,
{
    match data.len() {
        0..2 => (0..data.len()).map(|i| i.try_into().unwrap()).collect(),
        2 => {
            if data[0] < data[1] {
                vec![IndexType::zero(), IndexType::one()]
            } else {
                vec![IndexType::one(), IndexType::zero()]
            }
        }
        3..10 => suffix_array_naive(data),
        10..40 => suffix_array_doubling(data),
        _ => suffix_array_induced_sorting(data, alphabet_max),
    }
}

pub(crate) fn suffix_array(data: &[u32]) -> Vec<usize> {
    fn to_usize_compress(data: &[u32]) -> Vec<usize> {
        let mut unique_values = data.to_vec();
        unique_values.par_sort();
        unique_values.dedup();

        let value_index = unique_values
            .into_iter()
            .enumerate()
            .map(|(index, value)| (value, index))
            .collect::<std::collections::HashMap<_, _>>();

        data.iter().map(|opt| value_index[opt]).collect::<Vec<_>>()
    }

    if data.len() <= u8::MAX as usize {
        let data_usize = to_usize_compress(data)
            .into_iter()
            .map(|x| x as u8)
            .collect::<Vec<_>>();
        let &alphabet_max = data_usize.iter().max().unwrap_or(&0) as &u8;
        suffix_array_inner(data_usize, alphabet_max)
            .into_iter()
            .map(|x| x as usize)
            .collect()
    } else if data.len() <= u16::MAX as usize {
        let data_usize = to_usize_compress(data)
            .into_iter()
            .map(|x| x as u16)
            .collect::<Vec<_>>();
        let &alphabet_max = data_usize.iter().max().unwrap_or(&0) as &u16;
        suffix_array_inner(data_usize, alphabet_max)
            .into_iter()
            .map(|x| x as usize)
            .collect()
    } else if data.len() <= u32::MAX as usize {
        let data_usize = data.to_vec();
        let &alphabet_max = data_usize.iter().max().unwrap_or(&0) as &u32;
        suffix_array_inner(data_usize, alphabet_max)
            .into_iter()
            .map(|x| x as usize)
            .collect()
    } else {
        let data_usize = data.iter().map(|&x| x as u64).collect::<Vec<_>>();
        let &alphabet_max = data_usize.iter().max().unwrap_or(&0) as &u64;
        suffix_array_inner(data_usize, alphabet_max)
            .into_iter()
            .map(|x| x as usize)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    fn verify_all(array: &[usize], expected_idx: &[usize]) {
        let suffix_idx_doubling = suffix_array_doubling(array.to_vec());
        assert_eq!(suffix_idx_doubling, expected_idx);
        let suffix_idx_naive = suffix_array_naive(array.to_vec());
        assert_eq!(suffix_idx_naive, expected_idx);
        let suffix_idx_induced_sorting =
            suffix_array_induced_sorting(array.to_vec(), array.iter().copied().max().unwrap_or(0));
        assert_eq!(suffix_idx_induced_sorting, expected_idx);
        let suffix_idx =
            suffix_array_inner(array.to_vec(), array.iter().copied().max().unwrap_or(0));
        assert_eq!(suffix_idx, expected_idx);
        let suffix_idx_optional =
            suffix_array(&array.iter().map(|&x| x as u32).collect::<Vec<_>>());
        assert_eq!(suffix_idx_optional, expected_idx);
    }

    #[test]
    fn test_suffix_array_0() {
        let array = [0, 1, 2, 3, 4, 5];
        let suffix_idx = suffix_array_doubling(array.to_vec());
        assert_eq!(suffix_idx, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_suffix_array_1() {
        let str = "abracadabra";
        let array = str.bytes().map(|byte| byte as usize).collect::<Vec<_>>();
        verify_all(&array, &[10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2]);
    }

    #[test]
    fn test_suffix_array_2() {
        let str = "mmiissiissiippii"; // an example taken from https://mametter.hatenablog.com/entry/20180130/p1
        let array = str.bytes().map(|byte| byte as usize).collect::<Vec<_>>();
        verify_all(
            &array,
            &[15, 14, 10, 6, 2, 11, 7, 3, 1, 0, 13, 12, 9, 5, 8, 4],
        );
    }

    #[test]
    fn test_suffix_array_3() {
        let str = "mississippi".repeat(50);
        let array = str.bytes().map(|byte| byte as usize).collect::<Vec<_>>();
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
        assert_eq!(
            suffix_array(&array),
            &[
                37, 13, 6, 25, 5, 24, 21, 14, 17, 19, 3, 1, 7, 9, 11, 0, 22, 15, 18, 20, 36, 33,
                30, 27, 26, 4, 2, 8, 10, 35, 34, 23, 16, 12, 32, 29, 31, 28,
            ],
        );
    }
}
