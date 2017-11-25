#![feature(test)]

extern crate test;
extern crate rand;
extern crate rand_core;

use std::mem::{size_of, size_of_val};
use test::{black_box, Bencher};
use rand::{Rng, thread_rng, random};
use rand::hash::*;

#[bench]
fn hash_u64(b: &mut Bencher) {
    const N: u64 = 100;
    let mut x: u64 = random();
    b.iter(|| {
        for _ in 0..N {
            x = x.wrapping_add(1);  // unique number each time
            
            // Using `hash` instead of `hash_u64` here has big performance impact — why?
            black_box(SeaHash::new_fixed().hash_u64(x));
            //black_box(SeaHash::new_fixed().hash(x));
            //black_box(Hashable::hash(SeaHash::new_fixed(), x));
        }
    });
    b.bytes = size_of::<u64>() as u64 * N;
}

#[bench]
fn hash_bytes32(b: &mut Bencher) {
    const N: u64 = 20;
    let mut buf = [0u8; 32];
    thread_rng().fill_bytes(&mut buf);
    b.iter(|| {
        for _ in 0..N {
            buf[0] = buf[0].wrapping_add(1);  // unique number each time
            black_box(SeaHash::hash_fixed(&buf[..]));
        }
    });
    b.bytes = size_of_val(&buf) as u64 * N;
}

#[bench]
fn hash_bytes1024(b: &mut Bencher) {
    const N: u64 = 1;
    let mut buf = [0u8; 1024];
    thread_rng().fill_bytes(&mut buf);
    b.iter(|| {
        for _ in 0..N {
            buf[0] = buf[0].wrapping_add(1);  // unique number each time
            black_box(SeaHash::hash_fixed(&buf[..]));
        }
    });
    b.bytes = size_of_val(&buf) as u64 * N;
}
