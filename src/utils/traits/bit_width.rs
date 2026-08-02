use num_traits::Zero;

pub(crate) trait BitWidth {
    fn bit_width(&self) -> usize;
}

macro_rules! impl_bit_width_for_prim {
    ($($t:ty),*) => {
        $(
        impl BitWidth for $t {
            fn bit_width(&self) -> usize {
                if self.is_zero() {
                    0usize
                } else {
                    self.ilog2() as usize + 1
                }
            }
        }
        )*
    };
}

impl_bit_width_for_prim!(u32, u64, u128, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_width_prim() {
        let x: u32 = 18;
        assert_eq!(BitWidth::bit_width(&x), 5);
        let y: u64 = 0;
        assert_eq!(BitWidth::bit_width(&y), 0);
    }
}
