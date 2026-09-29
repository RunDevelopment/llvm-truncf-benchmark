use fearless_simd::{Sse2, f32x4, f64x2, i32x4, i64x2, prelude::*};
use fearless_simd_macros::simd;

#[inline]
pub fn floor_sse2_f32(x: f32) -> f32 {
    // Clear the sign bit
    let abs = x.abs();

    // Determine which values to return unchanged
    const THRESHOLD: f32 = 2147483648.0;
    let return_floor = abs <= THRESHOLD;

    // Truncate using the native SSE2 f32 <-> i32 conversions
    let abs_trunc = (abs as i32) as f32;
    let trunc = abs_trunc.copysign(x);

    // make sure the resulting integer is less than or equal to the original value
    let inc = trunc > x;
    let floor = if inc { trunc - 1.0 } else { trunc };

    if return_floor { floor } else { x }
}

#[inline]
#[simd]
pub fn floor_sse2_f32x4(x: f32x4<Sse2>) -> f32x4<Sse2> {
    // Clear the sign bit
    let abs = x.abs();

    // Determine which values to return unchanged
    const THRESHOLD: f32 = 2147483648.0;
    let return_floor = abs.simd_le(THRESHOLD);

    // Truncate using the native SSE2 f32 <-> i32 conversions
    let abs_trunc: f32x4<Sse2> = i32x4::truncate_from(abs).to_float();
    let trunc = abs_trunc.copysign(x);

    // make sure the resulting integer is less than or equal to the original value
    let inc = trunc.simd_gt(x);
    let floor = inc.select(trunc - 1.0, trunc);

    return_floor.select(floor, x)
}

#[inline]
pub fn floor_sse2_f64(x: f64) -> f64 {
    // Clear the sign bit
    let abs = x.abs();

    // Determine which values to return unchanged
    const THRESHOLD: f64 = 9223372036854775808.0;
    let return_floor = abs <= THRESHOLD;

    // Truncate using the native SSE2 f64 <-> i64 conversions
    let abs_trunc = (abs as i64) as f64;
    let trunc = abs_trunc.copysign(x);

    // make sure the resulting integer is less than or equal to the original value
    let inc = trunc > x;
    let floor = if inc { trunc - 1.0 } else { trunc };

    if return_floor { floor } else { x }
}

#[inline]
#[simd]
pub fn floor_sse2_f64x2(x: f64x2<Sse2>) -> f64x2<Sse2> {
    // Clear the sign bit
    let abs = x.abs();

    // Determine which values to return unchanged
    const THRESHOLD: f64 = 9223372036854775808.0;
    let return_floor = abs.simd_le(THRESHOLD);

    // Truncate using the native SSE2 f64 <-> i64 conversions
    let abs_trunc: f64x2<Sse2> = i64x2::truncate_from(abs).to_float();
    let trunc = abs_trunc.copysign(x);

    // make sure the resulting integer is less than or equal to the original value
    let inc = trunc.simd_gt(x);
    let floor = inc.select(trunc - 1.0, trunc);

    return_floor.select(floor, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fearless_simd::Level;

    #[test]
    fn proof_correctness_floor_sse2_f32() {
        // verify results are bit exact
        for bits in 0..=u32::MAX {
            let input = f32::from_bits(bits);
            let expected = input.floor();
            let actual = floor_sse2_f32(input);
            // the only exception to bit-exact are NaN values
            if input.is_nan() {
                assert!(expected.is_nan());
            } else {
                assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
            }
        }
    }

    #[test]
    fn proof_correctness_floor_sse2_f32x4() {
        let sse2 = Level::new().as_sse2().expect("SSE2 not available");

        // verify results are bit exact
        for bits in 0..=u32::MAX {
            let input = f32::from_bits(bits);
            let expected = input.floor();
            let actual = floor_sse2_f32x4(f32x4::splat(sse2, input))[0];
            // the only exception to bit-exact are NaN values
            if input.is_nan() {
                assert!(expected.is_nan());
            } else {
                assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
            }
        }
    }

    #[test]
    fn test_correctness_floor_sse2_f64() {
        crate::iter_u64s(16, 8, |bits| {
            let input = f64::from_bits(bits);
            let expected = input.floor();
            let actual = floor_sse2_f64(input);
            // the only exception to bit-exact are NaN values
            if input.is_nan() {
                assert!(expected.is_nan());
            } else {
                assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
            }
        });
    }

    #[test]
    fn test_correctness_floor_sse2_f64x2() {
        let sse2 = Level::new().as_sse2().expect("SSE2 not available");

        crate::iter_u64s(16, 8, |bits| {
            let input = f64::from_bits(bits);
            let expected = input.floor();
            let actual = floor_sse2_f64x2(f64x2::splat(sse2, input))[0];
            // the only exception to bit-exact are NaN values
            if input.is_nan() {
                assert!(expected.is_nan());
            } else {
                assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
            }
        });
    }
}
