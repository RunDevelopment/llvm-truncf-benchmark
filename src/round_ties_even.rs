use fearless_simd::{Sse2, f32x4, f64x2, prelude::*};
use fearless_simd_macros::simd;

const BIAS_F32: f32 = 8388608.0; // 2^23
const BIAS_F64: f64 = 4503599627370496.0; // 2^52

#[inline]
pub fn round_ties_even_sse2_f32(x: f32) -> f32 {
    let abs = x.abs();
    let bias = BIAS_F32;
    let res = ((abs + bias) - bias).copysign(x);
    if abs < bias { res } else { x }
}

#[inline]
#[simd]
pub fn round_ties_even_sse2_f32x4(x: f32x4<Sse2>) -> f32x4<Sse2> {
    let abs = x.abs();
    let bias = f32x4::splat(x.simd, BIAS_F32);
    let mask = abs.simd_lt(bias);
    let res = ((abs + bias) - bias).copysign(x);
    mask.select(res, x)
}

#[inline]
pub fn round_ties_even_sse2_f64(x: f64) -> f64 {
    let abs = x.abs();
    let bias = BIAS_F64;
    let res = ((abs + bias) - bias).copysign(x);
    if abs < bias { res } else { x }
}

#[inline]
#[simd]
pub fn round_ties_even_sse2_f64x2(x: f64x2<Sse2>) -> f64x2<Sse2> {
    let abs = x.abs();
    let bias = f64x2::splat(x.simd, BIAS_F64);
    let mask = abs.simd_lt(bias);
    let res = ((abs + bias) - bias).copysign(x);
    mask.select(res, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fearless_simd::Level;

    #[test]
    fn proof_correctness_round_ties_even_sse2_f32() {
        // verify results are bit exact
        for bits in 0..=u32::MAX {
            let input = f32::from_bits(bits);
            let expected = input.round_ties_even();
            let actual = round_ties_even_sse2_f32(input);
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
    }

    #[test]
    fn proof_correctness_round_ties_even_sse2_f32x4() {
        let sse2 = Level::new().as_sse2().expect("SSE2 not available");

        // verify results are bit exact
        for bits in 0..=u32::MAX {
            let input = f32::from_bits(bits);
            let expected = input.round_ties_even();
            let actual = round_ties_even_sse2_f32x4(f32x4::splat(sse2, input))[0];
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
    }

    #[test]
    fn test_correctness_round_ties_even_sse2_f64() {
        crate::iter_u64s(16, 8, |bits| {
            let input = f64::from_bits(bits);
            let expected = input.round_ties_even();
            let actual = round_ties_even_sse2_f64(input);
            assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
        });
    }

    #[test]
    fn test_correctness_round_ties_even_sse2_f64x2() {
        let sse2 = Level::new().as_sse2().expect("SSE2 not available");

        crate::iter_u64s(16, 8, |bits| {
            let input = f64::from_bits(bits);
            let expected = input.round_ties_even();
            let actual = round_ties_even_sse2_f64x2(f64x2::splat(sse2, input))[0];
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "{input} ({bits:x} {input}_f64)"
            );
        });
    }
}
