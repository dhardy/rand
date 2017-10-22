#![feature(test)]

extern crate test;
extern crate rand;

const RAND_BENCH_N: u64 = 1000;
const BYTES_LEN: usize = 1024;

use std::mem::size_of;
use test::{black_box, Bencher};

use rand::{Rng, NewSeeded, SeedFromRng, StdRng, OsRng, Rand, Default};
use rand::prng::{XorShiftRng, IsaacRng, Isaac64Rng, ChaChaRng};

macro_rules! gen_bytes {
    ($fnn:ident, $gen:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $gen::new().unwrap();
            let mut buf = [0u8; BYTES_LEN];
            b.iter(|| {
                for _ in 0..RAND_BENCH_N {
                    rng.try_fill(&mut buf).unwrap();
                    black_box(buf);
                }
            });
            b.bytes = BYTES_LEN as u64 * RAND_BENCH_N;
        }
    }
}

gen_bytes!(gen_bytes_xorshift, XorShiftRng);
gen_bytes!(gen_bytes_isaac, IsaacRng);
gen_bytes!(gen_bytes_isaac64, Isaac64Rng);
gen_bytes!(gen_bytes_chacha, ChaChaRng);
gen_bytes!(gen_bytes_std, StdRng);
gen_bytes!(gen_bytes_os, OsRng);

macro_rules! gen_u32 {
    ($fnn:ident, $gen:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $gen::new().unwrap();
            b.iter(|| {
                for _ in 0..RAND_BENCH_N {
                    black_box(u32::rand(&mut rng, Default));
                }
            });
            b.bytes = size_of::<u32>() as u64 * RAND_BENCH_N;
        }
    }
}

gen_u32!(gen_u32_xorshift, XorShiftRng);
gen_u32!(gen_u32_isaac, IsaacRng);
gen_u32!(gen_u32_isaac64, Isaac64Rng);
gen_u32!(gen_u32_chacha, ChaChaRng);
gen_u32!(gen_u32_std, StdRng);
gen_u32!(gen_u32_os, OsRng);

macro_rules! gen_u64 {
    ($fnn:ident, $gen:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $gen::new().unwrap();
            b.iter(|| {
                for _ in 0..RAND_BENCH_N {
                    black_box(u64::rand(&mut rng, Default));
                }
            });
            b.bytes = size_of::<u64>() as u64 * RAND_BENCH_N;
        }
    }
}

gen_u64!(gen_u64_xorshift, XorShiftRng);
gen_u64!(gen_u64_isaac, IsaacRng);
gen_u64!(gen_u64_isaac64, Isaac64Rng);
gen_u64!(gen_u64_chacha, ChaChaRng);
gen_u64!(gen_u64_std, StdRng);
gen_u64!(gen_u64_os, OsRng);

macro_rules! init_gen {
    ($fnn:ident, $gen:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = XorShiftRng::new().unwrap();
            b.iter(|| {
                for _ in 0..RAND_BENCH_N {
                    black_box($gen::from_rng(&mut rng).unwrap());
                }
            });
        }
    }
}

init_gen!(init_xorshift, XorShiftRng);
init_gen!(init_isaac, IsaacRng);
init_gen!(init_isaac64, Isaac64Rng);
init_gen!(init_chacha, ChaChaRng);
init_gen!(init_std, StdRng);
