#![feature(test)]
#![allow(non_snake_case)]

extern crate test;
extern crate rand;

use test::Bencher;

use rand::prelude::*;
use rand::seq::*;

#[bench]
fn seq_shuffle_100(b: &mut Bencher) {
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
    let x : &mut [usize] = &mut [1; 100];
    b.iter(|| {
        x.shuffle(&mut rng);
        x[0]
    })
}

#[bench]
fn seq_slice_choose_1_of_1000(b: &mut Bencher) {
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
    let x : &[usize] = &[1; 1000];
    b.iter(|| {
        x.choose(&mut rng)
    })
}

macro_rules! seq_slice_choose_multiple {
    ($name:ident, $amount:expr, $length:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
            let x : &[i32] = &[$amount; $length];
            let mut result = [0i32; $amount];
            b.iter(|| {
                for (slot, sample) in result.iter_mut().zip(
                    x.choose_multiple(&mut rng, $amount, false)) {
                    *slot = *sample;
                }
                result[$amount-1]
            })
        }
    }
}

seq_slice_choose_multiple!(seq_slice_choose_multiple_1_of_1000, 1, 1000);
seq_slice_choose_multiple!(seq_slice_choose_multiple_950_of_1000, 950, 1000);
seq_slice_choose_multiple!(seq_slice_choose_multiple_10_of_100, 10, 100);
seq_slice_choose_multiple!(seq_slice_choose_multiple_90_of_100, 90, 100);
seq_slice_choose_multiple!(seq_slice_choose_multiple_1k_of_1M, 1000, 1000_000);

#[bench]
fn seq_iter_choose_from_100(b: &mut Bencher) {
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
    let x : &[usize] = &[1; 100];
    b.iter(|| {
        x.iter().cloned().choose(&mut rng)
    })
}

#[bench]
fn seq_iter_choose_multiple_10_of_100(b: &mut Bencher) {
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
    let x : &[usize] = &[1; 100];
    b.iter(|| {
        x.iter().cloned().choose_multiple(&mut rng, 10)
    })
}

#[bench]
fn seq_iter_choose_multiple_fill_10_of_100(b: &mut Bencher) {
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
    let x : &[usize] = &[1; 100];
    let mut buf = [0; 10];
    b.iter(|| {
        x.iter().cloned().choose_multiple_fill(&mut rng, &mut buf)
    })
}

macro_rules! sample_indices {
    ($name:ident, $fn:ident, $amount:expr, $length:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
            b.iter(|| {
                $fn(&mut rng, $length, $amount, false).collect::<Vec<_>>()
            })
        }
    }
}

sample_indices!(misc_sample_indices_1_of_1k, sample_indices, 1, 1000);
sample_indices!(misc_sample_indices_10_of_1k, sample_indices, 10, 1000);
sample_indices!(misc_sample_indices_100_of_1k, sample_indices, 100, 1000);
sample_indices!(misc_sample_indices_100_of_1M, sample_indices, 100, 1000_000);
sample_indices!(misc_sample_indices_100_of_1G, sample_indices, 100, 1000_000_000);
sample_indices!(misc_sample_indices_400_of_1G, sample_indices, 400, 1000_000_000);
sample_indices!(misc_sample_indices_600_of_1G, sample_indices, 600, 1000_000_000);
