// Adapted from: https://github.com/rust-lang-ja/ac-library-rs/blob/0cdbc5e2ad110b688b0239e0208e275dde94a1e2/src/string.rs
use std::{cmp, iter, mem};

use num_traits::PrimInt;

fn suffix_array_naive(data: Vec<usize>) -> Vec<usize> {
    let length = data.len();

    let mut suffix_indices = (0..length).collect::<Vec<_>>();
    suffix_indices.sort_by(|&(mut left_pos), &(mut right_pos)| {
        if left_pos == right_pos {
            return cmp::Ordering::Equal;
        }
        while left_pos < length && right_pos < length {
            if data[left_pos] != data[right_pos] {
                return data[left_pos].cmp(&data[right_pos]);
            }
            left_pos += 1;
            right_pos += 1;
        }
        if left_pos == length {
            cmp::Ordering::Less
        } else {
            cmp::Ordering::Greater
        }
    });

    suffix_indices
}

fn suffix_array_doubling(data: Vec<usize>) -> Vec<usize> {
    let length = data.len();

    let mut suffix_indices = (0..length).collect::<Vec<_>>();
    let mut current_rank = data;

    let mut next_rank = vec![0usize; length];
    let mut prefix_len = 1;

    while prefix_len < length {
        let compare_suffix = |&pos_i: &usize, &pos_j: &usize| {
            if current_rank[pos_i] != current_rank[pos_j] {
                return current_rank[pos_i].cmp(&current_rank[pos_j]);
            }
            match (pos_i + prefix_len < length, pos_j + prefix_len < length) {
                (false, false) => cmp::Ordering::Equal,
                (false, true) => cmp::Ordering::Less,
                (true, false) => cmp::Ordering::Greater,
                (true, true) => {
                    current_rank[pos_i + prefix_len].cmp(&current_rank[pos_j + prefix_len])
                }
            }
        };

        suffix_indices.sort_by(compare_suffix);

        next_rank[suffix_indices[0]] = 0;
        for i in 1..length {
            let should_increment =
                compare_suffix(&suffix_indices[i - 1], &suffix_indices[i]) == cmp::Ordering::Less;
            next_rank[suffix_indices[i]] = if should_increment {
                next_rank[suffix_indices[i - 1]] + 1
            } else {
                next_rank[suffix_indices[i - 1]]
            };
        }

        mem::swap(&mut current_rank, &mut next_rank);
        prefix_len *= 2;
    }

    suffix_indices
}

fn suffix_array_induced_sorting(data: Vec<usize>, alphabet_max: usize) -> Vec<usize> {
    let length = data.len();
    let mut suffix_indices = vec![0; length];
    let mut is_s_type = vec![false; length];
    for i in (0..length - 1).rev() {
        is_s_type[i] = if data[i] == data[i + 1] {
            is_s_type[i + 1]
        } else {
            data[i] < data[i + 1]
        };
    }

    let mut bucket_l_start = vec![0; alphabet_max + 1];
    let mut bucket_s_start = vec![0; alphabet_max + 1];
    for (&is_s_type, &data) in iter::zip(&is_s_type, &data) {
        if is_s_type {
            bucket_l_start[data + 1] += 1;
        } else {
            bucket_s_start[data] += 1;
        }
    }
    for i in 0..=alphabet_max {
        bucket_s_start[i] += bucket_l_start[i];
        if i < alphabet_max {
            bucket_l_start[i + 1] += bucket_s_start[i];
        }
    }

    // suffix array's origin is +1
    let induced_sort = |suffix_indices: &mut Vec<usize>, lms_positions: &Vec<usize>| {
        suffix_indices.iter_mut().for_each(|elem| {
            *elem = 0;
        });
        let mut bucket_cursor = bucket_s_start.clone();
        for &lms_pos in lms_positions {
            if lms_pos == length {
                continue;
            }
            let insert_pos = bucket_cursor[data[lms_pos]];
            bucket_cursor[data[lms_pos]] += 1;
            suffix_indices[insert_pos] = lms_pos + 1;
        }
        bucket_cursor.copy_from_slice(&bucket_l_start);
        let insert_pos = bucket_cursor[data[length - 1]];
        bucket_cursor[data[length - 1]] += 1;
        suffix_indices[insert_pos] = length;
        for i in 0..length {
            let suffix_value = suffix_indices[i];
            if suffix_value >= 2 && !is_s_type[suffix_value - 2] {
                let insert_pos = bucket_cursor[data[suffix_value - 2]];
                bucket_cursor[data[suffix_value - 2]] += 1;
                suffix_indices[insert_pos] = suffix_value - 1;
            }
        }
        bucket_cursor.copy_from_slice(&bucket_l_start);
        for i in (0..length).rev() {
            let suffix_value = suffix_indices[i];
            if suffix_value >= 2 && is_s_type[suffix_value - 2] {
                bucket_cursor[data[suffix_value - 2] + 1] -= 1;
                let insert_pos = bucket_cursor[data[suffix_value - 2] + 1];
                suffix_indices[insert_pos] = suffix_value - 1;
            }
        }
    };

    // origin of lms_index is +1
    let mut lms_index = vec![0usize; length + 1];
    let mut num_lms = 0usize;
    for i in 1..length {
        if !is_s_type[i - 1] && is_s_type[i] {
            lms_index[i] = num_lms + 1;
            num_lms += 1;
        }
    }
    let lms_positions = (1..length)
        .filter(|&i| !is_s_type[i - 1] && is_s_type[i])
        .collect::<Vec<_>>();
    debug_assert_eq!(lms_positions.len(), num_lms);
    induced_sort(&mut suffix_indices, &lms_positions);

    if num_lms > 0 {
        let mut sorted_lms_positions = suffix_indices
            .iter()
            .filter_map(|&suffix_value| {
                if lms_index[suffix_value - 1] > 0 {
                    Some(suffix_value - 1)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut reduced_data = vec![0usize; num_lms];
        let mut reduced_alphabet_max = 0usize;
        reduced_data[lms_index[sorted_lms_positions[0]] - 1] = 0usize;
        for i in 1..num_lms {
            let mut prev_pos = sorted_lms_positions[i - 1];
            let mut curr_pos = sorted_lms_positions[i];
            let prev_end = if lms_index[prev_pos] < num_lms {
                lms_positions[lms_index[prev_pos]]
            } else {
                length
            };
            let curr_end = if lms_index[curr_pos] < num_lms {
                lms_positions[lms_index[curr_pos]]
            } else {
                length
            };
            let is_same_lms_substring = if prev_end - prev_pos != curr_end - curr_pos {
                false
            } else {
                while prev_pos < prev_end && data[prev_pos] == data[curr_pos] {
                    prev_pos += 1;
                    curr_pos += 1;
                }
                prev_pos != length && data[prev_pos] == data[curr_pos]
            };

            if !is_same_lms_substring {
                reduced_alphabet_max += 1;
            }
            reduced_data[lms_index[sorted_lms_positions[i]] - 1] = reduced_alphabet_max;
        }

        let reduced_suffix_array = suffix_array(reduced_data, reduced_alphabet_max);
        for i in 0..num_lms {
            sorted_lms_positions[i] = lms_positions[reduced_suffix_array[i]];
        }
        induced_sort(&mut suffix_indices, &sorted_lms_positions);
    }
    suffix_indices.iter_mut().for_each(|x| *x -= 1);
    suffix_indices
}

fn suffix_array(data: Vec<usize>, alphabet_max: usize) -> Vec<usize> {
    match data.len() {
        0..2 => (0..data.len()).collect(),
        2 => {
            if data[0] < data[1] {
                vec![0, 1]
            } else {
                vec![1, 0]
            }
        }
        3..10 => suffix_array_naive(data),
        10..40 => suffix_array_doubling(data),
        _ => suffix_array_induced_sorting(data, alphabet_max),
    }
}

pub(crate) fn suffix_array_option<Element: PrimInt>(data: &[Option<Element>]) -> Vec<usize> {
    let data = data
        .iter()
        .map(|opt| match opt {
            Some(value) => value.to_usize().unwrap() + 1,
            None => 0,
        })
        .collect::<Vec<_>>();
    let &alphabet_max = data.iter().max().unwrap_or(&0);
    suffix_array(data, alphabet_max)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    fn verify_all_algorithms(array: &[usize], expected_indices: &[usize]) {
        let result_doubling = suffix_array_doubling(array.to_vec());
        assert_eq!(result_doubling, expected_indices);

        let result_naive = suffix_array_naive(array.to_vec());
        assert_eq!(result_naive, expected_indices);

        let result_induced_sorting =
            suffix_array_induced_sorting(array.to_vec(), array.iter().copied().max().unwrap_or(0));
        assert_eq!(result_induced_sorting, expected_indices);

        let result_auto = suffix_array(array.to_vec(), array.iter().copied().max().unwrap_or(0));
        assert_eq!(result_auto, expected_indices);

        let result_option =
            suffix_array_option(&array.iter().map(|&x| Some(x)).collect::<Vec<_>>());
        assert_eq!(result_option, expected_indices);
    }

    #[test]
    fn test_sorted_sequence() {
        let array = [0, 1, 2, 3, 4, 5];
        let result = suffix_array_doubling(array.to_vec());
        assert_eq!(result, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_abracadabra() {
        let text = "abracadabra";
        let array = text.bytes().map(|byte| byte as usize).collect::<Vec<_>>();
        verify_all_algorithms(&array, &[10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2]);
    }

    #[test]
    fn test_repeated_characters() {
        // Example from https://mametter.hatenablog.com/entry/20180130/p1
        let text = "mmiissiissiippii";
        let array = text.bytes().map(|byte| byte as usize).collect::<Vec<_>>();
        verify_all_algorithms(
            &array,
            &[15, 14, 10, 6, 2, 11, 7, 3, 1, 0, 13, 12, 9, 5, 8, 4],
        );
    }

    #[test]
    fn test_long_repeated_text() {
        let text = "mississippi".repeat(50);
        let array = text.bytes().map(|byte| byte as usize).collect::<Vec<_>>();
        verify_all_algorithms(
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
    fn test_multiple_strings_with_separators() {
        let texts = ["banana", "ananas", "abracadabra", "mississippi"];
        let array = texts
            .iter()
            .flat_map(|&text| text.bytes().map(Some).chain(iter::once(None)))
            .collect::<Vec<_>>();

        let expected = vec![
            37, 13, 6, 25, 5, 24, 21, 14, 17, 19, 3, 1, 7, 9, 11, 0, 22, 15, 18, 20, 36, 33, 30,
            27, 26, 4, 2, 8, 10, 35, 34, 23, 16, 12, 32, 29, 31, 28,
        ];
        assert_eq!(suffix_array_option(&array), expected);
    }
}
