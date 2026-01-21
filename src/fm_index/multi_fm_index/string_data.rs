#[derive(PartialEq, PartialOrd, Eq, Ord)]
pub(crate) enum StringData {
    Ucs1(Vec<u8>),
    Ucs2(Vec<u16>),
    Ucs4(Vec<u32>),
}
