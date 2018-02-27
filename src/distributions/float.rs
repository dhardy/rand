// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Basic floating-point number distributions

use {RngCore};
use distributions::{Distribution, Uniform};
use utils::FloatConversions;


/// A distribution to sample floating point numbers uniformly in the open
/// interval `(0, 1)` (not including either endpoint).
///
/// See also: [`Closed01`] for the closed `[0, 1]`; [`Uniform`] for the
/// half-open `[0, 1)`.
///
/// # Example
/// ```rust
/// use rand::{NewRng, SmallRng, Rng};
/// use rand::distributions::Open01;
///
/// let val: f32 = SmallRng::new().sample(Open01);
/// println!("f32 from (0,1): {}", val);
/// ```
///
/// [`Uniform`]: struct.Uniform.html
/// [`Closed01`]: struct.Closed01.html
#[derive(Clone, Copy, Debug)]
pub struct Open01;

/// A distribution to sample floating point numbers uniformly in the closed
/// interval `[0, 1]` (including both endpoints).
///
/// See also: [`Open01`] for the open `(0, 1)`; [`Uniform`] for the half-open
/// `[0, 1)`.
///
/// # Example
/// ```rust
/// use rand::{thread_rng, Rng};
/// use rand::distributions::Closed01;
///
/// let val: f32 = thread_rng().sample(Closed01);
/// println!("f32 from [0,1]: {}", val);
/// ```
///
/// [`Uniform`]: struct.Uniform.html
/// [`Open01`]: struct.Open01.html
#[derive(Clone, Copy, Debug)]
pub struct Closed01;


macro_rules! float_impls {
    ($ty:ty, $next_u:path) => {
        impl Distribution<$ty> for Uniform {
            fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> $ty {
                let x = $next_u(rng);
                x.closed_open01()
            }
        }

        impl Distribution<$ty> for Open01 {
            // Sample from the open range (0,1).
            // This uses the rejection method: use `closed_open01`, and if the
            // result is 0.0, reject the result and try again.
            fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> $ty {
                let mut x = 0;
                while x == 0 { // 0 converts to 0.0
                    x = $next_u(rng);
                }
                x.closed_open01()
            }
        }

        impl Distribution<$ty> for Closed01 {
            fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> $ty {
                // The problem with a closed range over [0,1] is that it needs
                // 2^n+1 samples to generate a uniform distribution. Which is
                // difficult, as it means we either have to reject about half
                // the generated random numbers, or just not include one number
                // in the distribution. That is why, instead of a closed range,
                // we actually sample from the half-open range (0,1].
                //
                // Floating-point numbers have more precision near zero. This
                // means for a f32 that only 1 in 2^32 samples will give exactly
                // 0.0. But 1 in 2^23 will give exactly 1.0 due to rounding.
                // Because the chance to sample 0.0 is so low, this half-open
                // range is a very good appoximation of a closed range.
                let x = $next_u(rng);
                x.open_closed01()
            }
        }
    }
}
float_impls! { f32, RngCore::next_u32 }
float_impls! { f64, RngCore::next_u64 }


#[cfg(test)]
mod tests {
    use {Rng};
    use distributions::{Open01, Closed01};
    use mock::StepRng;

    const EPSILON32: f32 = ::core::f32::EPSILON;
    const EPSILON64: f64 = ::core::f64::EPSILON;

    #[test]
    fn floating_point_edge_cases() {
        let mut zeros = StepRng::new(0, 0);
        assert_eq!(zeros.gen::<f32>(), 0.0);
        assert_eq!(zeros.gen::<f64>(), 0.0);
        
        /* TODO: update target values or remove
        let mut one = StepRng::new(1, 0);
        assert_eq!(one.gen::<f32>(), EPSILON32);
        assert_eq!(one.gen::<f64>(), EPSILON64);
        
        let mut max = StepRng::new(!0, 0);
        assert_eq!(max.gen::<f32>(), 1.0 - EPSILON32);
        assert_eq!(max.gen::<f64>(), 1.0 - EPSILON64);
        */
    }

    #[test]
    fn fp_closed_edge_cases() {
        /* TODO: update target values or remove
        let mut zeros = StepRng::new(0, 0);
        assert_eq!(zeros.sample::<f32, _>(Closed01), 0.0);
        assert_eq!(zeros.sample::<f64, _>(Closed01), 0.0);
        
        let mut one = StepRng::new(1, 0);
        let one32 = one.sample::<f32, _>(Closed01);
        let one64 = one.sample::<f64, _>(Closed01);
        assert!(EPSILON32 < one32 && one32 < EPSILON32 * 1.01);
        assert!(EPSILON64 < one64 && one64 < EPSILON64 * 1.01);
        
        let mut max = StepRng::new(!0, 0);
        assert_eq!(max.sample::<f32, _>(Closed01), 1.0);
        assert_eq!(max.sample::<f64, _>(Closed01), 1.0);
        */
    }

    #[test]
    fn fp_open_edge_cases() {
        /* We cannot do this test: it resamples until generated num is not 0.
        let mut zeros = StepRng::new(0, 0);
        assert_eq!(zeros.sample::<f32, _>(Open01), 0.0 + EPSILON32 / 2.0);
        assert_eq!(zeros.sample::<f64, _>(Open01), 0.0 + EPSILON64 / 2.0);
        */
        
        /* TODO: update target values or remove
        let mut one = StepRng::new(1, 0);
        let one32 = one.sample::<f32, _>(Open01);
        let one64 = one.sample::<f64, _>(Open01);
        assert!(EPSILON32 < one32 && one32 < EPSILON32 * 2.0);
        assert!(EPSILON64 < one64 && one64 < EPSILON64 * 2.0);
        */
        
        let mut max = StepRng::new(!0, 0);
        assert_eq!(max.sample::<f32, _>(Open01), 1.0 - EPSILON32 / 2.0);
        assert_eq!(max.sample::<f64, _>(Open01), 1.0 - EPSILON64 / 2.0);
    }

    #[test]
    fn rand_open() {
        // this is unlikely to catch an incorrect implementation that
        // generates exactly 0 or 1, but it keeps it sane.
        let mut rng = ::test::rng(510);
        for _ in 0..1_000 {
            // strict inequalities
            let f: f64 = rng.sample(Open01);
            assert!(0.0 < f && f < 1.0);

            let f: f32 = rng.sample(Open01);
            assert!(0.0 < f && f < 1.0);
        }
    }

    #[test]
    fn rand_closed() {
        let mut rng = ::test::rng(511);
        for _ in 0..1_000 {
            // strict inequalities
            let f: f64 = rng.sample(Closed01);
            assert!(0.0 <= f && f <= 1.0);

            let f: f32 = rng.sample(Closed01);
            assert!(0.0 <= f && f <= 1.0);
        }
    }
}
