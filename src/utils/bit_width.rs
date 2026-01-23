use num_traits::Zero;

pub(super) fn bit_width(value: u32) -> usize {
    if value.is_zero() {
        0usize
    } else {
        value.ilog2() as usize + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_width_prim() {
        let x = 18u32;
        assert_eq!(bit_width(x), 5);
        let y = 0u32;
        assert_eq!(bit_width(y), 0);
    }
}
