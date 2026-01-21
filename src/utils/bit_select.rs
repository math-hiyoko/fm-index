use num_traits::Zero;

pub(super) trait BitSelect {
    /// Select the position of the k-th set bit (1-based index).
    fn bit_select(&self, bit: bool, k: usize) -> Option<usize>;
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "bmi2")]
unsafe fn pdep_select_u32_bmi2(value: u32, kth: usize) -> Option<usize> {
    use core::arch::x86::_pdep_u32;

    if kth.is_zero() {
        return None;
    }
    if (value.count_ones() as usize) < kth {
        return None;
    }

    Some(_pdep_u32(1u32 << (kth - 1), value).trailing_zeros() as usize)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn pdep_select_u64_bmi2(value: u64, kth: usize) -> Option<usize> {
    use core::arch::x86_64::_pdep_u64;

    if kth.is_zero() {
        return None;
    }
    if (value.count_ones() as usize) < kth {
        return None;
    }

    Some(_pdep_u64(1u64 << (kth - 1), value).trailing_zeros() as usize)
}

#[inline]
fn fallback_select_u32(value: u32, mut kth: usize) -> Option<usize> {
    if kth.is_zero() {
        return None;
    }
    if (value.count_ones() as usize) < kth {
        return None;
    }

    let x1 = value - ((value & 0xAAAAAAAAu32) >> 1);
    let x2 = (x1 & 0x33333333u32) + ((x1 >> 2) & 0x33333333u32);
    let x3 = (x2 + (x2 >> 4)) & 0x0F0F0F0Fu32;

    let mut pos = 0;
    loop {
        let cnt = ((x3 >> pos) & 0xFFu32) as usize;
        if kth <= cnt {
            break;
        }
        kth -= cnt;
        pos += 8;
    }

    let cnt4 = ((x2 >> pos) & 0x0Fu32) as usize;
    if kth > cnt4 {
        kth -= cnt4;
        pos += 4;
    }

    let cnt2 = ((x1 >> pos) & 0x03u32) as usize;
    if kth > cnt2 {
        kth -= cnt2;
        pos += 2;
    }

    let bit0 = ((value >> pos) & 1u32) as usize;
    if bit0 < kth {
        pos += 1;
    }

    Some(pos)
}

#[inline]
fn fallback_select_u64(value: u64, mut kth: usize) -> Option<usize> {
    if kth.is_zero() {
        return None;
    }
    if (value.count_ones() as usize) < kth {
        return None;
    }

    let x1 = value - ((value & 0xAAAAAAAAAAAAAAAAu64) >> 1);
    let x2 = (x1 & 0x3333333333333333u64) + ((x1 >> 2) & 0x3333333333333333u64);
    let x3 = (x2 + (x2 >> 4)) & 0x0F0F0F0F0F0F0F0Fu64;

    let mut pos = 0;
    loop {
        let cnt = ((x3 >> pos) & 0xFFu64) as usize;
        if kth <= cnt {
            break;
        }
        kth -= cnt;
        pos += 8;
    }

    let cnt4 = ((x2 >> pos) & 0x0Fu64) as usize;
    if kth > cnt4 {
        kth -= cnt4;
        pos += 4;
    }

    let cnt2 = ((x1 >> pos) & 0x03u64) as usize;
    if kth > cnt2 {
        kth -= cnt2;
        pos += 2;
    }

    let bit0 = ((value >> pos) & 1u64) as usize;
    if bit0 < kth {
        pos += 1;
    }

    Some(pos)
}

impl BitSelect for u32 {
    fn bit_select(&self, bit: bool, kth: usize) -> Option<usize> {
        let value = if bit { *self } else { !*self };

        #[cfg(target_arch = "x86")]
        {
            if std::arch::is_x86_feature_detected!("bmi2") {
                return unsafe { pdep_select_u32_bmi2(value, kth) };
            }
        }

        fallback_select_u32(value, kth)
    }
}

impl BitSelect for u64 {
    fn bit_select(&self, bit: bool, kth: usize) -> Option<usize> {
        let value = if bit { *self } else { !*self };

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("bmi2") {
                return unsafe { pdep_select_u64_bmi2(value, kth) };
            }
        }

        fallback_select_u64(value, kth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_select_u32() {
        let value = 0b10110010010110001011001001011000u32;

        // Select with kth=0 should return None
        assert_eq!(value.bit_select(true, 0), None);
        assert_eq!(value.bit_select(false, 0), None);

        // Valid selections
        assert_eq!(value.bit_select(true, 6), Some(13));
        assert_eq!(value.bit_select(false, 15), Some(24));

        // Out of range selections
        assert_eq!(value.bit_select(true, 15), None);
        assert_eq!(value.bit_select(false, 19), None);

        // Edge case: all zeros
        let zero = 0u32;
        assert_eq!(zero.bit_select(true, 1), None);
        assert_eq!(zero.bit_select(false, 1), Some(0));
    }

    #[test]
    fn test_bit_select_u64() {
        let value = 0b1011001001011000101100100101100010110010010110001011001001011000u64;

        // Select with kth=0 should return None
        assert_eq!(value.bit_select(true, 0), None);
        assert_eq!(value.bit_select(false, 0), None);

        // Valid selections
        assert_eq!(value.bit_select(true, 20), Some(45));
        assert_eq!(value.bit_select(false, 25), Some(42));

        // Out of range selections
        assert_eq!(value.bit_select(true, 29), None);
        assert_eq!(value.bit_select(false, 37), None);

        // Edge case: all zeros
        let zero = 0u64;
        assert_eq!(zero.bit_select(true, 1), None);
        assert_eq!(zero.bit_select(false, 1), Some(0));
    }
}
