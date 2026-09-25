use fearless_simd::{Sse2, f32x4, f64x2, i32x4, i64x2, prelude::*};
use fearless_simd_macros::simd;

#[inline]
pub fn trunc_sse2_f32(x: f32) -> f32 {
    // Clear the sign bit
    let abs = x.abs();

    // Determine which values to return unchanged
    const INT_THRESHOLD: i32 = f32::to_bits(2147483648.0) as i32;
    let return_unchanged = abs.to_bits() as i32 >= INT_THRESHOLD;

    // Truncate using the native SSE2 f32 <-> i32 conversions
    let abs_trunc = (abs as i32) as f32;
    let trunc = abs_trunc.copysign(x);

    if return_unchanged { x } else { trunc }
}

#[inline]
#[simd]
pub fn trunc_sse2_f32x4(x: f32x4<Sse2>) -> f32x4<Sse2> {
    // Clear the sign bit
    let abs = x.abs();

    // Determine which values to return unchanged
    const INT_THRESHOLD: i32 = f32::to_bits(2147483648.0) as i32;
    let return_unchanged = abs
        .bitcast::<i32x4<Sse2>>()
        .simd_ge(i32x4::splat(x.simd, INT_THRESHOLD));

    // Truncate using the native SSE2 f32 <-> i32 conversions
    let abs_trunc: f32x4<Sse2> = i32x4::truncate_from(abs).to_float();
    let trunc = abs_trunc.copysign(x);

    return_unchanged.select(x, trunc)
}

#[inline]
pub fn trunc_sse2_f64(x: f64) -> f64 {
    // Clear the sign bit
    let abs = x.abs();

    // Determine which values to return unchanged
    const INT_THRESHOLD: i64 = f64::to_bits(9223372036854775808.0) as i64;
    let return_unchanged = abs.to_bits() as i64 >= INT_THRESHOLD;

    // Truncate using the native SSE2 f64 <-> i64 conversions
    let abs_trunc = (abs as i64) as f64;
    let trunc = abs_trunc.copysign(x);

    if return_unchanged { x } else { trunc }
}

#[inline]
#[simd]
pub fn trunc_sse2_f64x2(x: f64x2<Sse2>) -> f64x2<Sse2> {
    // Clear the sign bit
    let abs = x.abs();

    // Determine which values to return unchanged
    const INT_THRESHOLD: i64 = f64::to_bits(9223372036854775808.0) as i64;
    let return_unchanged = abs
        .bitcast::<i64x2<Sse2>>()
        .simd_ge(i64x2::splat(x.simd, INT_THRESHOLD));

    // Truncate using the native SSE2 f64 <-> i64 conversions
    let abs_trunc: f64x2<Sse2> = i64x2::truncate_from(abs).to_float();
    let trunc = abs_trunc.copysign(x);

    return_unchanged.select(x, trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fearless_simd::Level;

    #[test]
    fn proof_correctness_trunc_sse2_f32() {
        // verify results are bit exact
        for bits in 0..=u32::MAX {
            let input = f32::from_bits(bits);
            let expected = input.trunc();
            let actual = trunc_sse2_f32(input);
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
    }

    #[test]
    fn proof_correctness_trunc_sse2_f32x4() {
        let sse2 = Level::new().as_sse2().expect("SSE2 not available");

        // verify results are bit exact
        for bits in 0..=u32::MAX {
            let input = f32::from_bits(bits);
            let expected = input.trunc();
            let actual = trunc_sse2_f32x4(f32x4::splat(sse2, input))[0];
            assert_eq!(expected.to_bits(), actual.to_bits());
        }
    }

    #[test]
    fn test_correctness_trunc_sse2_f64() {
        crate::iter_u64s(16, 8, |bits| {
            let input = f64::from_bits(bits);
            let expected = input.trunc();
            let actual = trunc_sse2_f64(input);
            assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
        });
    }

    #[test]
    fn test_correctness_trunc_sse2_f64x2() {
        let sse2 = Level::new().as_sse2().expect("SSE2 not available");

        crate::iter_u64s(16, 8, |bits| {
            let input = f64::from_bits(bits);
            let expected = input.trunc();
            let actual = trunc_sse2_f64x2(f64x2::splat(sse2, input))[0];
            assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
        });
    }
}
