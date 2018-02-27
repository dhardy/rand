// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

//! The Gamma and derived distributions.

use self::GammaRepr::*;
use self::ChiSquaredRepr::*;

use {Rng, RngCore};
use distributions::normal::StandardNormal;
use distributions::{Distribution, Exp, Open01};

/// The Gamma distribution `Gamma(shape, scale)` distribution.
///
/// The density function of this distribution is
///
/// ```text
/// f(x) =  x^(k - 1) * exp(-x / θ) / (Γ(k) * θ^k)
/// ```
///
/// where `Γ` is the Gamma function, `k` is the shape and `θ` is the
/// scale and both `k` and `θ` are strictly positive.
///
/// The algorithm used is that described by Marsaglia & Tsang 2000[1],
/// falling back to directly sampling from an Exponential for `shape
/// == 1`, and using the boosting technique described in [1] for
/// `shape < 1`.
///
/// # Example
///
/// ```rust
/// use rand::distributions::{Distribution, Gamma};
///
/// let gamma = Gamma::new(2.0, 5.0);
/// let v = gamma.sample(&mut rand::thread_rng());
/// println!("{} is from a Gamma(2, 5) distribution", v);
/// ```
///
/// [1]: George Marsaglia and Wai Wan Tsang. 2000. "A Simple Method
/// for Generating Gamma Variables" *ACM Trans. Math. Softw.* 26, 3
/// (September 2000),
/// 363-372. DOI:[10.1145/358407.358414](https://doi.acm.org/10.1145/358407.358414)
#[derive(Clone, Copy, Debug)]
pub struct Gamma {
    repr: GammaRepr,
}

#[derive(Clone, Copy, Debug)]
enum GammaRepr {
    Large(GammaLargeShape),
    One(Exp),
    Small(GammaSmallShape)
}

// These two helpers could be made public, but saving the
// match-on-Gamma-enum branch from using them directly (e.g. if one
// knows that the shape is always > 1) doesn't appear to be much
// faster.

/// Gamma distribution where the shape parameter is less than 1.
///
/// Note, samples from this require a compulsory floating-point `pow`
/// call, which makes it significantly slower than sampling from a
/// gamma distribution where the shape parameter is greater than or
/// equal to 1.
///
/// See `Gamma` for sampling from a Gamma distribution with general
/// shape parameters.
#[derive(Clone, Copy, Debug)]
struct GammaSmallShape {
    inv_shape: f64,
    large_shape: GammaLargeShape
}

/// Gamma distribution where the shape parameter is larger than 1.
///
/// See `Gamma` for sampling from a Gamma distribution with general
/// shape parameters.
#[derive(Clone, Copy, Debug)]
struct GammaLargeShape {
    scale: f64,
    c: f64,
    d: f64
}

impl Gamma {
    /// Construct an object representing the `Gamma(shape, scale)`
    /// distribution.
    ///
    /// Panics if `shape <= 0` or `scale <= 0`.
    #[inline]
    pub fn new(shape: f64, scale: f64) -> Gamma {
        assert!(shape > 0.0, "Gamma::new called with shape <= 0");
        assert!(scale > 0.0, "Gamma::new called with scale <= 0");

        let repr = if shape == 1.0 {
            One(Exp::new(1.0 / scale))
        } else if shape < 1.0 {
            Small(GammaSmallShape::new_raw(shape, scale))
        } else {
            Large(GammaLargeShape::new_raw(shape, scale))
        };
        Gamma { repr: repr }
    }
}

impl GammaSmallShape {
    fn new_raw(shape: f64, scale: f64) -> GammaSmallShape {
        GammaSmallShape {
            inv_shape: 1. / shape,
            large_shape: GammaLargeShape::new_raw(shape + 1.0, scale)
        }
    }
}

impl GammaLargeShape {
    fn new_raw(shape: f64, scale: f64) -> GammaLargeShape {
        let d = shape - 1. / 3.;
        GammaLargeShape {
            scale: scale,
            c: 1. / (9. * d).sqrt(),
            d: d
        }
    }
}

impl Distribution<f64> for Gamma {
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        match self.repr {
            Small(ref g) => g.sample(rng),
            One(ref g) => g.sample(rng),
            Large(ref g) => g.sample(rng),
        }
    }
}
impl Distribution<f64> for GammaSmallShape {
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.sample(Open01);

        self.large_shape.sample(rng) * u.powf(self.inv_shape)
    }
}
impl Distribution<f64> for GammaLargeShape {
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let x = rng.sample(StandardNormal);
            let v_cbrt = 1.0 + self.c * x;
            if v_cbrt <= 0.0 { // a^3 <= 0 iff a <= 0
                continue
            }

            let v = v_cbrt * v_cbrt * v_cbrt;
            let u: f64 = rng.sample(Open01);

            let x_sqr = x * x;
            if u < 1.0 - 0.0331 * x_sqr * x_sqr ||
                u.ln() < 0.5 * x_sqr + self.d * (1.0 - v + v.ln()) {
                return self.d * v * self.scale
            }
        }
    }
}

/// The chi-squared distribution `χ²(k)`, where `k` is the degrees of
/// freedom.
///
/// For `k > 0` integral, this distribution is the sum of the squares
/// of `k` independent standard normal random variables. For other
/// `k`, this uses the equivalent characterisation
/// `χ²(k) = Gamma(k/2, 2)`.
///
/// # Example
///
/// ```rust
/// use rand::distributions::{ChiSquared, Distribution};
///
/// let chi = ChiSquared::new(11.0);
/// let v = chi.sample(&mut rand::thread_rng());
/// println!("{} is from a χ²(11) distribution", v)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ChiSquared {
    repr: ChiSquaredRepr,
}

#[derive(Clone, Copy, Debug)]
enum ChiSquaredRepr {
    // k == 1, Gamma(alpha, ..) is particularly slow for alpha < 1,
    // e.g. when alpha = 1/2 as it would be for this case, so special-
    // casing and using the definition of N(0,1)^2 is faster.
    DoFExactlyOne,
    DoFAnythingElse(Gamma),
}

impl ChiSquared {
    /// Create a new chi-squared distribution with degrees-of-freedom
    /// `k`. Panics if `k < 0`.
    pub fn new(k: f64) -> ChiSquared {
        let repr = if k == 1.0 {
            DoFExactlyOne
        } else {
            assert!(k > 0.0, "ChiSquared::new called with `k` < 0");
            DoFAnythingElse(Gamma::new(0.5 * k, 2.0))
        };
        ChiSquared { repr: repr }
    }
}
impl Distribution<f64> for ChiSquared {
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        match self.repr {
            DoFExactlyOne => {
                // k == 1 => N(0,1)^2
                let norm = rng.sample(StandardNormal);
                norm * norm
            }
            DoFAnythingElse(ref g) => g.sample(rng)
        }
    }
}

/// The Fisher F distribution `F(m, n)`.
///
/// This distribution is equivalent to the ratio of two normalised
/// chi-squared distributions, that is, `F(m,n) = (χ²(m)/m) /
/// (χ²(n)/n)`.
///
/// # Example
///
/// ```rust
/// use rand::distributions::{FisherF, Distribution};
///
/// let f = FisherF::new(2.0, 32.0);
/// let v = f.sample(&mut rand::thread_rng());
/// println!("{} is from an F(2, 32) distribution", v)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FisherF {
    numer: ChiSquared,
    denom: ChiSquared,
    // denom_dof / numer_dof so that this can just be a straight
    // multiplication, rather than a division.
    dof_ratio: f64,
}

impl FisherF {
    /// Create a new `FisherF` distribution, with the given
    /// parameter. Panics if either `m` or `n` are not positive.
    pub fn new(m: f64, n: f64) -> FisherF {
        assert!(m > 0.0, "FisherF::new called with `m < 0`");
        assert!(n > 0.0, "FisherF::new called with `n < 0`");

        FisherF {
            numer: ChiSquared::new(m),
            denom: ChiSquared::new(n),
            dof_ratio: n / m
        }
    }
}
impl Distribution<f64> for FisherF {
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        self.numer.sample(rng) / self.denom.sample(rng) * self.dof_ratio
    }
}

/// The Student t distribution, `t(nu)`, where `nu` is the degrees of
/// freedom.
///
/// # Example
///
/// ```rust
/// use rand::distributions::{StudentT, Distribution};
///
/// let t = StudentT::new(11.0);
/// let v = t.sample(&mut rand::thread_rng());
/// println!("{} is from a t(11) distribution", v)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StudentT {
    chi: ChiSquared,
    dof: f64
}

impl StudentT {
    /// Create a new Student t distribution with `n` degrees of
    /// freedom. Panics if `n <= 0`.
    pub fn new(n: f64) -> StudentT {
        assert!(n > 0.0, "StudentT::new called with `n <= 0`");
        StudentT {
            chi: ChiSquared::new(n),
            dof: n
        }
    }
}
impl Distribution<f64> for StudentT {
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        let norm = rng.sample(StandardNormal);
        norm * (self.dof / self.chi.sample(rng)).sqrt()
    }
}

#[cfg(test)]
mod test {
    use distributions::Distribution;
    use super::{ChiSquared, StudentT, FisherF};

    #[test]
    fn test_chi_squared_one() {
        let chi = ChiSquared::new(1.0);
        let mut rng = ::test::rng(201);
        for _ in 0..1000 {
            chi.sample(&mut rng);
        }
    }
    #[test]
    fn test_chi_squared_small() {
        let chi = ChiSquared::new(0.5);
        let mut rng = ::test::rng(202);
        for _ in 0..1000 {
            chi.sample(&mut rng);
        }
    }
    #[test]
    fn test_chi_squared_large() {
        let chi = ChiSquared::new(30.0);
        let mut rng = ::test::rng(203);
        for _ in 0..1000 {
            chi.sample(&mut rng);
        }
    }
    #[test]
    #[should_panic]
    fn test_chi_squared_invalid_dof() {
        ChiSquared::new(-1.0);
    }

    #[test]
    fn test_f() {
        let f = FisherF::new(2.0, 32.0);
        let mut rng = ::test::rng(204);
        for _ in 0..1000 {
            f.sample(&mut rng);
        }
    }

    #[test]
    fn test_t() {
        let t = StudentT::new(11.0);
        let mut rng = ::test::rng(205);
        for _ in 0..1000 {
            t.sample(&mut rng);
        }
    }
}
