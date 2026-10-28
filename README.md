# Truncation micro benchmark

This repo implements multiple micro benchmark to measure the performance of truncation and rounding algorithms.

The basic setup is to apply truncation to all floats in a buffer and write the result to an output buffer.
This tests both the performance of the operation itself and how well it vectorizes.

I also provide variants which are hand-vectorizes using `fearless_simd`. This can be thought of as the ceiling to aim for.

## Usage

Required software: Rust >= 1.87.0
Requires hardware: Any x86 chip with SSE2 support (=any Intel/Amd chip released within the last 20 years)

The following command will compile the program with optimizations and run the benchmark:

```bash
cargo run --release
```

It will run the micro benchmarks and output a table like this:

```
name                         256         512        1024        2048        4096        8192       16384

f32::trunc               1.334µs     2.581µs     5.326µs    11.991µs    27.519µs    58.341µs   122.308µs
trunc_sse2_f32             246ns       483ns       950ns     1.887µs     3.772µs     7.521µs     15.02µs
trunc_sse2_f32x4            60ns       120ns       236ns       468ns       923ns     1.838µs     3.665µs

f64::trunc               1.054µs     2.038µs     4.082µs     9.597µs     21.74µs    45.048µs   112.369µs
trunc_sse2_f64             309ns       599ns      1.18µs     2.671µs    11.113µs    28.828µs    62.701µs
trunc_sse2_f64x2           233ns       460ns       900ns      1.79µs     3.583µs     7.126µs    14.264µs

f32::round                2.33µs     4.744µs    10.027µs    14.682µs    32.246µs    68.615µs   139.843µs
round_sse2_f32             276ns       548ns     1.079µs     2.147µs     4.295µs     8.628µs     17.06µs
round_sse2_f32x4            69ns       139ns       273ns       534ns     1.095µs     2.116µs     4.197µs

f64::round               2.483µs     5.091µs    10.374µs    13.683µs    29.301µs    61.317µs    125.24µs
round_sse2_f64             343ns       652ns     1.336µs      2.98µs    12.223µs    32.332µs    72.655µs
round_sse2_f64x2           259ns       504ns     1.003µs     1.996µs     3.987µs     7.938µs    15.892µs

...
```

- The left-most column is the name of the function benchmarked.
- The numbers in the heading say how many floats are being processed per iteration.
- The measurements are mean iteration time of the method applied the number of floats.

The above numbers were measured on the following system:

- CPU: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
- OS: MS Windows 10.0.19045.7725
- Rust 1.98.1

## Code and testing

```
Cargo.toml    # project config
src/
    main.r s  # contains the benchmarking code
    floor.rs  # floor implementations
    ceil.rs   # ceil implementations
    trunc.rs  # truncation implementations
    round.rs  # rounding implementations
    round_ties_even.rs  # round ties even implementations
```

All algorithms are tested for correctness. Tests for float32 algorithms serve as proofs for correctness by virtue of testing all possible inputs against the reference implementation.

Run all tests using `cargo test`.

Note: Correctness proof take around a minute per test.

## License

All code and text in this repo is licensed under CC-BY-4.0 OR MIT License OR Apache 2.0 at your choosing.
