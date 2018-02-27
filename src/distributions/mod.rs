// Copyright 2013-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Sampling from random distributions.
//!
//! A distribution may have internal state describing the distribution of
//! generated values; for example `Range` needs to know its upper and lower
//! bounds. Distributions use the `Distribution` trait to yield values: call
//! `distr.sample(&mut rng)` to get a random variable.
use {Rng, RngCore};

pub use self::float::{Open01, Closed01};
pub use self::other::Alphanumeric;
pub use self::range::Range;
#[cfg(feature="std")]
pub use self::gamma::{Gamma, ChiSquared, FisherF, StudentT};
#[cfg(feature="std")]
pub use self::normal::{Normal, LogNormal, StandardNormal};
#[cfg(feature="std")]
pub use self::exponential::{Exp, Exp1};
#[cfg(feature = "std")]
pub use self::poisson::Poisson;
#[cfg(feature = "std")]
pub use self::binomial::Binomial;

pub mod range;

#[cfg(feature="std")]
pub mod gamma;
#[cfg(feature="std")]
pub mod normal;
#[cfg(feature="std")]
pub mod exponential;
#[cfg(feature = "std")]
pub mod poisson;
#[cfg(feature = "std")]
pub mod binomial;

mod float;
mod integer;
#[cfg(feature="std")]
mod log_gamma;
mod other;
#[cfg(feature="std")]
mod ziggurat_tables;

/// Types (distributions) that can be used to create a random instance of `T`.
pub trait Distribution<T> {
    /// Generate a random value of `T`, using `rng` as the
    /// source of randomness.
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T;
}

impl<'a, T, D: Distribution<T>> Distribution<T> for &'a D {
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        (*self).sample(rng)
    }
}

/// A generic random value distribution. Generates values for various types
/// with numerically uniform distribution.
/// 
/// For floating-point numbers, this generates values from the open range
/// `(0, 1)` (i.e. excluding 0.0 and 1.0).
///
/// ## Built-in Implementations
///
/// This crate implements the distribution `Uniform` for various primitive
/// types.  Assuming the provided `Rng` is well-behaved, these implementations
/// generate values with the following ranges and distributions:
///
/// * Integers (`i32`, `u32`, `isize`, `usize`, etc.): Uniformly distributed
///   over all values of the type.
/// * `char`: Uniformly distributed over all Unicode scalar values, i.e. all
///   code points in the range `0...0x10_FFFF`, except for the range
///   `0xD800...0xDFFF` (the surrogate code points). This includes
///   unassigned/reserved code points.
/// * `bool`: Generates `false` or `true`, each with probability 0.5.
/// * Floating point types (`f32` and `f64`): Uniformly distributed in the
///   open range `(0, 1)`.
///
/// The following aggregate types also implement the distribution `Uniform` as
/// long as their component types implement it:
///
/// * Tuples and arrays: Each element of the tuple or array is generated
///   independently, using the `Uniform` distribution recursively.
/// * `Option<T>`: Returns `None` with probability 0.5; otherwise generates a
///   random `T` and returns `Some(T)`.
///
/// # Example
/// ```rust
/// use rand::{NewRng, SmallRng, Rng};
/// use rand::distributions::Uniform;
///
/// let val: f32 = SmallRng::new().sample(Uniform);
/// println!("f32 from [0,1): {}", val);
/// ```
///
/// With dynamic dispatch (type erasure of `Rng`):
/// 
/// ```rust
/// use rand::{thread_rng, Rng, RngCore};
/// use rand::distributions::Uniform;
///
/// let mut rng = thread_rng();
/// let mut erased_rng: &mut RngCore = &mut rng;
/// let val: f32 = erased_rng.sample(Uniform);
/// println!("f32 from [0,1): {}", val);
/// ```
///
/// [`Exp1`]: struct.Exp1.html
/// [`StandardNormal`]: struct.StandardNormal.html
#[derive(Debug)]
pub struct Uniform;

/// Sample a random number using the Ziggurat method (specifically the
/// ZIGNOR variant from Doornik 2005). Most of the arguments are
/// directly from the paper:
///
/// * `rng`: source of randomness
/// * `symmetric`: whether this is a symmetric distribution, or one-sided with P(x < 0) = 0.
/// * `X`: the $x_i$ abscissae.
/// * `F`: precomputed values of the PDF at the $x_i$, (i.e. $f(x_i)$)
/// * `F_DIFF`: precomputed values of $f(x_i) - f(x_{i+1})$
/// * `pdf`: the probability density function
/// * `zero_case`: manual sampling from the tail when we chose the
///    bottom box (i.e. i == 0)

// the perf improvement (25-50%) is definitely worth the extra code
// size from force-inlining.
#[cfg(feature="std")]
#[inline(always)]
fn ziggurat<R: RngCore + ?Sized, P, Z>(
            rng: &mut R,
            symmetric: bool,
            x_tab: ziggurat_tables::ZigTable,
            f_tab: ziggurat_tables::ZigTable,
            mut pdf: P,
            mut zero_case: Z)
            -> f64 where P: FnMut(f64) -> f64, Z: FnMut(&mut R, f64) -> f64 {
    use utils::FloatConversions;
    
    loop {
        // As an optimisation convert the random u64 to a f64 using only
        // 53 bits, as many as will fit in the float's fraction.
        // Of the remaining 11 least significant bits we use 8 to construct `i`.
        // This saves us generating a whole extra random number, while the added
        // precision of using 64 bits for f64 does not buy us much.
        // Because for some RNG's the least significant bits can be of lower
        // statistical quality, we use bits 3..10 for i.
        let bits: u64 = rng.sample(Uniform);

        // u is either U(-1, 1) or U(0, 1) depending on if this is a
        // symmetric distribution or not.
        // FIXME: the distribution is not open, but closed-open.
        //        Can that cause problems or a bias?
        let u = if symmetric {
                    bits.closed_open11_fixed()
                } else {
                    bits.closed_open01_fixed()
                };
        let i = ((bits >> 3) & 0xff) as usize;

        let x = u * x_tab[i];

        let test_x = if symmetric { x.abs() } else {x};

        // algebraically equivalent to |u| < x_tab[i+1]/x_tab[i] (or u < x_tab[i+1]/x_tab[i])
        if test_x < x_tab[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        // algebraically equivalent to f1 + DRanU()*(f0 - f1) < 1
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * rng.gen::<f64>() < pdf(x) {
            return x;
        }
    }
}
