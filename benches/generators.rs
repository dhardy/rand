#![feature(test)]

extern crate test;
extern crate rand;

const RAND_BENCH_N: u64 = 1000;
const BYTES_LEN: usize = 1024;

use std::mem::size_of;
use test::{black_box, Bencher};

use rand::{Rng, NewSeeded, SeedFromRng, StdRng, ClockRng, StrongClockRng,
    OsRng, Rand, Default};
use rand::prng::{XorShiftRng, IsaacRng, Isaac64Rng, ChaChaRng};

macro_rules! gen_bytes {
    ($fnn:ident, $gen:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $gen;
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

gen_bytes!(gen_bytes_xorshift, XorShiftRng::try_new().unwrap());
gen_bytes!(gen_bytes_isaac, IsaacRng::try_new().unwrap());
gen_bytes!(gen_bytes_isaac64, Isaac64Rng::try_new().unwrap());
gen_bytes!(gen_bytes_chacha, ChaChaRng::try_new().unwrap());
gen_bytes!(gen_bytes_std, StdRng::try_new().unwrap());
gen_bytes!(gen_bytes_clock, ClockRng::new(2));
gen_bytes!(gen_bytes_strongclock, StrongClockRng::new());


macro_rules! gen_usize {
    ($fnn:ident, $gen:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $gen;
            b.iter(|| {
                for _ in 0..RAND_BENCH_N {
                    black_box(usize::rand(&mut rng, Default));
                }
            });
            b.bytes = size_of::<usize>() as u64 * RAND_BENCH_N;
        }
    }
}

gen_usize!(gen_usize_xorshift, XorShiftRng::try_new().unwrap());
gen_usize!(gen_usize_isaac, IsaacRng::try_new().unwrap());
gen_usize!(gen_usize_isaac64, Isaac64Rng::try_new().unwrap());
gen_usize!(gen_usize_chacha, ChaChaRng::try_new().unwrap());
gen_usize!(gen_usize_std, StdRng::try_new().unwrap());
gen_usize!(gen_usize_clock, ClockRng::new(2));
gen_usize!(gen_usize_os, OsRng::try_new().unwrap());
gen_usize!(gen_usize_strongclock, StrongClockRng::new());

macro_rules! init_gen {
    ($fnn:ident, $gen:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = XorShiftRng::try_new().unwrap();
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

// Differs from above in that it doesn't have a seeding rng
#[bench]
fn init_clock0(b: &mut Bencher) {
    b.iter(|| {
        for _ in 0..RAND_BENCH_N {
            black_box(ClockRng::new(0));
        }
    });
}
#[bench]
fn init_clock2(b: &mut Bencher) {
    b.iter(|| {
        for _ in 0..RAND_BENCH_N {
            black_box(ClockRng::new(2));
        }
    });
}
#[bench]
fn init_clock12(b: &mut Bencher) {
    b.iter(|| {
        for _ in 0..RAND_BENCH_N {
            black_box(ClockRng::new(12));
        }
    });
}
#[bench]
fn init_clock20(b: &mut Bencher) {
    b.iter(|| {
        for _ in 0..RAND_BENCH_N {
            black_box(ClockRng::new(20));
        }
    });
}
#[bench]
fn init_clock32(b: &mut Bencher) {
    b.iter(|| {
        for _ in 0..RAND_BENCH_N {
            black_box(ClockRng::new(32));
        }
    });
}
