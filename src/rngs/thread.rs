// Copyright 2018 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Thread-local random number generator

use core::cell::UnsafeCell;
use std::fmt;
use std::rc::Rc;
use std::thread_local;

use rand_core::{CryptoRng, RngCore, SeedableRng};

use super::std::Core;
use crate::rngs::OsRng;
use crate::rngs::ReseedingRng;

// Rationale for using `UnsafeCell` in `ThreadRng`:
//
// Previously we used a `RefCell`, with an overhead of ~15%. There will only
// ever be one mutable reference to the interior of the `UnsafeCell`, because
// we only have such a reference inside `next_u32`, `next_u64`, etc. Within a
// single thread (which is the definition of `ThreadRng`), there will only ever
// be one of these methods active at a time.
//
// A possible scenario where there could be multiple mutable references is if
// `ThreadRng` is used inside `next_u32` and co. But the implementation is
// completely under our control. We just have to ensure none of them use
// `ThreadRng` internally, which is nonsensical anyway. We should also never run
// `ThreadRng` in destructors of its implementation, which is also nonsensical.

// Number of generated bytes after which to reseed `ThreadRng`.
// According to benchmarks, reseeding has a noticeable impact with thresholds
// of 32 kB and less. We choose 64 kB to avoid significant overhead.
const THREAD_RNG_RESEED_THRESHOLD: u64 = 1024 * 64;

/// A reference to the thread-local generator
///
/// This type is a reference to a lazily-initialized thread-local generator.
/// An instance can be obtained via [`rand::rng()`][crate::rng())] or via
/// `ThreadRng::default()`.
/// The handle cannot be passed between threads (is not `Send` or `Sync`).
///
/// `ThreadRng` uses the same CSPRNG as [`StdRng`], ChaCha12. As with
/// [`StdRng`], the algorithm may be changed, subject to reasonable expectations
/// of security and performance.
///
/// `ThreadRng` is automatically seeded from [`OsRng`] with periodic reseeding
/// (every 64 kiB — see [`ReseedingRng`] documentation for details).
///
/// `ThreadRng` is not automatically reseeded on fork. It is recommended to
/// explicitly call [`ThreadRng::reseed`] immediately after a fork, for example:
/// ```ignore
/// fn do_fork() {
///     let pid = unsafe { libc::fork() };
///     if pid == 0 {
///         // Reseed ThreadRng in child processes:
///         rand::rng().reseed();
///     }
/// }
/// ```
///
/// Methods on `ThreadRng` are not reentrant-safe and thus should not be called
/// from an interrupt (e.g. a fork handler) unless it can be guaranteed that no
/// other method on the same `ThreadRng` is currently executing.
///
/// Security must be considered relative to a threat model and validation
/// requirements. `ThreadRng` attempts to meet basic security considerations
/// for producing unpredictable random numbers: use a CSPRNG, use a
/// recommended platform-specific seed ([`OsRng`]), and avoid
/// leaking internal secrets e.g. via [`Debug`] implementation or serialization.
/// Memory is not zeroized on drop.
///
/// [`ReseedingRng`]: crate::rngs::ReseedingRng
/// [`StdRng`]: crate::rngs::StdRng
#[derive(Clone)]
pub struct ThreadRng {
    // Rc is explicitly !Send and !Sync
    rng: Rc<UnsafeCell<ReseedingRng<Core, OsRng>>>,
}

impl ThreadRng {
    /// Immediately reseed the generator
    ///
    /// This discards any remaining random data in the cache.
    pub fn reseed(&mut self) -> Result<(), rand_core::getrandom::Error> {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.reseed()
    }
}

/// Debug implementation does not leak internal state
impl fmt::Debug for ThreadRng {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "ThreadRng {{ .. }}")
    }
}

thread_local!(
    // We require Rc<..> to avoid premature freeing when ThreadRng is used
    // within thread-local destructors. See #968.
    static THREAD_RNG_KEY: Rc<UnsafeCell<ReseedingRng<Core, OsRng>>> = {
        let r = Core::try_from_os_rng().unwrap_or_else(|err|
                panic!("could not initialize ThreadRng: {}", err));
        let rng = ReseedingRng::new(r,
                                    THREAD_RNG_RESEED_THRESHOLD,
                                    OsRng);
        Rc::new(UnsafeCell::new(rng))
    }
);

/// Access a local, pre-initialized generator
///
/// This is a reasonably fast unpredictable thread-local instance of [`ThreadRng`].
///
/// See also [`crate::rngs`] for alternatives.
///
/// # Example
///
/// ```
/// use rand::prelude::*;
///
/// # fn main() {
///
/// let mut numbers = [1, 2, 3, 4, 5];
/// numbers.shuffle(&mut rand::rng());
/// println!("Numbers: {numbers:?}");
///
/// // Using a local binding avoids an initialization-check on each usage:
/// let mut rng = rand::rng();
///
/// println!("True or false: {}", rng.random::<bool>());
/// println!("A simulated die roll: {}", rng.random_range(1..=6));
/// # }
/// ```
pub fn rng() -> ThreadRng {
    let rng = THREAD_RNG_KEY.with(|t| t.clone());
    ThreadRng { rng }
}

impl Default for ThreadRng {
    fn default() -> ThreadRng {
        rng()
    }
}

impl RngCore for ThreadRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u64()
    }

    #[inline(always)]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.fill_bytes(dest)
    }
}

impl CryptoRng for ThreadRng {}

#[cfg(test)]
mod test {
    #[test]
    fn test_thread_rng() {
        use crate::Rng;
        let mut r = crate::rng();
        r.random::<i32>();
        assert_eq!(r.random_range(0..1), 0);
    }

    #[test]
    fn test_debug_output() {
        // We don't care about the exact output here, but it must not include
        // private CSPRNG state or the cache stored by BlockRng!
        assert_eq!(std::format!("{:?}", crate::rng()), "ThreadRng { .. }");
    }
}
