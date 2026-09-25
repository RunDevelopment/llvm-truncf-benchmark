mod round;
mod trunc;

use std::{hint::black_box, io::Write, time::Duration};

use fearless_simd::{Level, Sse2, f32x4, f64x2, prelude::*};
use rand::prelude::*;

use round::*;
use trunc::*;

fn main() {
    let mut data_f32 = vec![0.0f32; 16384];
    let mut data_f64 = vec![0.0f64; 16384];

    // initialize data to random bits
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(123456789);
    for x in &mut data_f32 {
        *x = f32::from_bits(rng.next_u32());
    }
    for x in &mut data_f64 {
        *x = f64::from_bits(rng.next_u64());
    }

    bench_headers();
    println!();
    bench("f32::trunc", &data_f32, scalar(f32::trunc));
    bench("trunc_sse2_f32", &data_f32, scalar(trunc_sse2_f32));
    bench("trunc_sse2_f32x4", &data_f32, sse2_f32x4(trunc_sse2_f32x4));
    println!();
    bench("f64::trunc", &data_f64, scalar(f64::trunc));
    bench("trunc_sse2_f64", &data_f64, scalar(trunc_sse2_f64));
    bench("trunc_sse2_f64x2", &data_f64, sse2_f64x2(trunc_sse2_f64x2));
    println!();
    bench("f32::round", &data_f32, scalar(f32::round));
    bench("round_sse2_f32", &data_f32, scalar(round_sse2_f32));
    bench("round_sse2_f32x4", &data_f32, sse2_f32x4(round_sse2_f32x4));
    println!();
    bench("f64::round", &data_f64, scalar(f64::round));
    bench("round_sse2_f64", &data_f64, scalar(round_sse2_f64));
    bench("round_sse2_f64x2", &data_f64, sse2_f64x2(round_sse2_f64x2));
}

const WARM_UP: u32 = 100;
const ITERS: u32 = 100_000;
const MAX_DURATION: Duration = Duration::from_millis(2_000);
const LENGTHS: &[usize] = &[256, 512, 1024, 2048, 4096, 8192, 16384];
fn bench<T: Copy>(name: &str, data: &[T], f: impl Fn(&[T], &mut [T])) {
    print!("{name:<20}");
    for &len in LENGTHS {
        // prepare input and output buffers
        let data = &data[..len];
        let mut output = data.to_vec();
        let output = output.as_mut_slice();

        // a few warmup iterations
        for _ in 0..black_box(WARM_UP) {
            #[allow(clippy::unit_arg)]
            black_box(f(black_box(data), black_box(output)));
        }

        let batch_size = 1000;
        let mut i = 0;
        let start = std::time::Instant::now();
        while i < ITERS && start.elapsed() < MAX_DURATION {
            for _ in 0..black_box(batch_size) {
                #[allow(clippy::unit_arg)]
                black_box(f(black_box(data), black_box(output)));
            }
            i += batch_size;
        }

        let elapsed = start.elapsed();
        print!("{:>12?}", elapsed / i);
        std::io::stdout().flush().unwrap();
    }
    println!();
}
fn bench_headers() {
    print!("{:<20}", "name");
    for &len in LENGTHS {
        print!("{:>12}", len);
    }
    println!();
}

fn scalar<T: Copy>(f: impl Fn(T) -> T) -> impl Fn(&[T], &mut [T]) {
    move |data, output| {
        for (i, o) in data.iter().zip(output.iter_mut()) {
            *o = f(*i);
        }
    }
}
fn sse2_f32x4(f: impl Fn(f32x4<Sse2>) -> f32x4<Sse2>) -> impl Fn(&[f32], &mut [f32]) {
    let sse2 = Level::new().as_sse2().expect("SSE2 not available");

    move |data, output| {
        let data = data.as_chunks::<4>().0;
        let output = output.as_chunks_mut::<4>().0;
        for (i, o) in data.iter().zip(output.iter_mut()) {
            *o = *f(f32x4::simd_from(sse2, *i));
        }
    }
}
fn sse2_f64x2(f: impl Fn(f64x2<Sse2>) -> f64x2<Sse2>) -> impl Fn(&[f64], &mut [f64]) {
    let sse2 = Level::new().as_sse2().expect("SSE2 not available");

    move |data, output| {
        let data = data.as_chunks::<2>().0;
        let output = output.as_chunks_mut::<2>().0;
        for (i, o) in data.iter().zip(output.iter_mut()) {
            *o = *f(f64x2::simd_from(sse2, *i));
        }
    }
}

/// Iterates a subset of all u64s by going through all possible combinations of high and low bits.
/// The bits between the high and low bits are filled with either 0 or 1.
#[allow(unused)]
pub(crate) fn iter_u64s(high_bits: u8, low_bits: u8, f: impl Fn(u64)) {
    for high in 0..(1 << high_bits) {
        for fill in 0..2 {
            let fill_mask = ((1 << (64 - high_bits - low_bits)) - 1) << low_bits;
            let fill = fill_mask * fill;
            for low in 0..(1 << low_bits) {
                let bits = (high << (64 - high_bits)) | fill | low;
                f(bits)
            }
        }
    }
}
