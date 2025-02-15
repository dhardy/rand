// Copyright 2025 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A cache over `OsRng`

use crate::rngs::OsRng;
use crate::Rng;
use core::fmt;
use rand_core::block::{BlockRng, BlockRngCore};
use rand_core::{RngCore, TryRngCore};

#[repr(transparent)]
struct Array64<T>([T; 64]);
impl<T> Default for Array64<T>
where
    T: Default,
{
    #[rustfmt::skip]
    fn default() -> Self {
        Self([
            T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(),
            T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(),
            T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(),
            T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(),
            T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(),
            T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(),
            T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(),
            T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(),
        ])
    }
}
impl<T> AsRef<[T]> for Array64<T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}
impl<T> AsMut<[T]> for Array64<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}
impl<T> Clone for Array64<T>
where
    T: Copy + Default,
{
    fn clone(&self) -> Self {
        let mut new = Self::default();
        new.0.copy_from_slice(&self.0);
        new
    }
}

/// A cache over `OsRng`
#[derive(Default)]
pub struct CachedOsRng(BlockRng<OsRngCore>);

impl fmt::Debug for CachedOsRng {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CachedOsRng {{ .. }}")
    }
}

impl CachedOsRng {
    /// Immediately discard any remaining random data in the cache
    pub fn reset(&mut self) {
        self.0.reset();
    }
}

#[derive(Default)]
struct OsRngCore;

impl BlockRngCore for OsRngCore {
    type Item = u32;
    type Results = Array64<u32>;

    #[inline]
    fn generate(&mut self, r: &mut Self::Results) {
        OsRng.unwrap_err().fill(r.as_mut());
    }
}

// TODO: we should implement TryRngCore instead? But then BlockRng needs to use that.
impl RngCore for CachedOsRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        if bytes.len() >= 256 {
            OsRng.unwrap_err().fill_bytes(bytes);
        } else {
            self.0.fill_bytes(bytes)
        }
    }
}

// CryptoRng
