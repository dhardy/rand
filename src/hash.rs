// Copyright 2017 The Rust Project Developers and Ticki.
// See the COPYRIGHT file at the top-level directory of this distribution and
// at http://rust-lang.org/COPYRIGHT.
// 
// A significant part of this work is copied with little modification from
// https://github.com/redox-os/tfs/tree/master/seahash
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides three things: a stable hash function, and a trait to
//! facilitate conversion of input to byte slices, and a trait allowing
//! multiple output sizes.

use core::slice;


/// A fixed seed for `SeaHash`. This is the same as used by `seahash::hash`.
/// You can use this seed or make your own.
pub const SEA_SEED: (u64, u64, u64, u64) = (0x16f11fe89b0d677c,
    0xb480a793d8e6c86c, 0x6fe2e5aaf078ebc9, 0x14f994a4c5259381);

/// An implementation of
/// [SeaHash](https://github.com/redox-os/tfs/tree/master/seahash).
/// 
/// This is essentially a copy of the linked implementation customised for our needs. Output is
/// equivalent, aside from our extensions to allow larger output state.
/// 
/// SeaHash is:
/// 
/// *   extremely fast
/// *   portable, i.e. reproducible across architectures and versions
/// *   well distributed
/// 
/// **This is not a cryptographic hash function,** despite the option for large output sizes. The
/// only uses for output larger than 64-bit are to reduce collisions (likely insignificant) and to
/// preserve more entropy (useful when used to produce a key, in cases where *both* input and
/// output of the hash function is secret). [To avoid confusion, a *cryptographic hash function*
/// should have the property that it is computationally infeasible to find an input producing a
/// known output. SeaHash does not have this property.]
/// 
/// Note that for larger output sizes, the output is essentially multiple 64-bit hashes taken on
/// different subsets of the input with no mixing between the hashes. In particular, if there is
/// less than 25 bytes of input, a subset of the output will simply be a copy of the input key
/// (reduced via XOR).
#[derive(Clone, Debug)]
pub struct SeaHash {
    /// `a`
    a: u64,
    /// `b`
    b: u64,
    /// `c`
    c: u64,
    /// `d`
    d: u64,
    /// The number of written bytes.
    written: u64,
}

// Hash a number up to 64 bits; cast to u64 and supply length in bytes
macro_rules! hash_num {
    ($self:ident, $input: expr, $len: expr) => {{
        let a = helper::diffuse($self.a ^ $input);
        SeaHash {
            a: a,
            b: $self.b,
            c: $self.c,
            d: $self.d,
            written: $self.written + $len,
        }
    }}
}

impl SeaHash {
    /// Create an instance with supplied seed
    pub fn new((a, b, c, d): (u64, u64, u64, u64)) -> Self {
        SeaHash{ a, b, c, d, written: 0 }
    }
    
    /// Create an instance with a fixed, default seed
    /// 
    /// This is equivalent to `SeaHash::new(SEA_SEED)`.
    pub fn new_fixed() -> Self {
        SeaHash::new(SEA_SEED)
    }
    
    /// Hash a value of any supported state to the given state.
    /// 
    /// Note that this consumes self instead of accepting `&mut self`; benchmarking shows a 30%
    /// performance reduction with the latter on `u64` input (though smaller at larger input);
    /// also returning self is allows expression-oriented usage.
    #[inline(always)]
    pub fn hash<T: Hashable>(self, input: T) -> Self {
        Hashable::hash(self, input)
    }
    
    /// Convenience version of `SeaHash::new_fixed().hash(input)`.
    #[inline(always)]
    pub fn hash_fixed<T: Hashable>(input: T) -> Self {
        Hashable::hash(SeaHash::new_fixed(), input)
    }
    
    /// Hash a buffer to the given state.
    pub fn hash_buf(self, buf: &[u8]) -> Self {
        let SeaHash { mut a, mut b, mut c, mut d, written } = self;
        unsafe {
            // We use 4 different registers to store seperate hash states, because this allows us
            // to update them seperately, and consequently exploiting ILP to update the states in
            // parallel.

            // The pointer to the current bytes.
            let mut ptr = buf.as_ptr();
            // The end of the "main segment", i.e. the biggest buffer s.t. the length is divisible
            // by 32.
            let end_ptr = buf.as_ptr().offset(buf.len() as isize & !0x1F);

            while end_ptr > ptr {
                // Modern CPUs allow the pointer arithmetic to be done in place, hence not
                // introducing tmpvars.
                a ^= helper::read_u64(ptr);
                b ^= helper::read_u64(ptr.offset(8));
                c ^= helper::read_u64(ptr.offset(16));
                d ^= helper::read_u64(ptr.offset(24));

                // Increment the pointer.
                ptr = ptr.offset(32);

                // Diffuse the updated registers. We hope that each of these are executed in
                // parallel.
                a = helper::diffuse(a);
                b = helper::diffuse(b);
                c = helper::diffuse(c);
                d = helper::diffuse(d);
            }

            // Calculate the number of excessive bytes. These are bytes that could not be handled
            // in the loop above.
            let mut excessive = buf.len() % 32;
            // Handle the excessive bytes.
            match excessive {
                0 => {},
                1...7 => {
                    // 1 or more excessive.

                    // Write the last excessive bytes (<8 bytes).
                    a ^= helper::read_int(slice::from_raw_parts(ptr, excessive));

                    // Diffuse.
                    a = helper::diffuse(a);
                },
                8 => {
                    // 8 bytes excessive.

                    // Mix in the partial block.
                    a ^= helper::read_u64(ptr);

                    // Diffuse.
                    a = helper::diffuse(a);
                },
                9...15 => {
                    // More than 8 bytes excessive.

                    // Mix in the partial block.
                    a ^= helper::read_u64(ptr);

                    // Write the last excessive bytes (<8 bytes).
                    excessive = excessive - 8;
                    b ^= helper::read_int(slice::from_raw_parts(ptr.offset(8), excessive));

                    // Diffuse.
                    a = helper::diffuse(a);
                    b = helper::diffuse(b);

                },
                16 => {
                    // 16 bytes excessive.

                    // Mix in the partial block.
                    a ^= helper::read_u64(ptr);
                    b ^= helper::read_u64(ptr.offset(8));

                    // Diffuse.
                    a = helper::diffuse(a);
                    b = helper::diffuse(b);
                },
                17...23 => {
                    // 16 bytes or more excessive.

                    // Mix in the partial block.
                    a ^= helper::read_u64(ptr);
                    b ^= helper::read_u64(ptr.offset(8));

                    // Write the last excessive bytes (<8 bytes).
                    excessive = excessive - 16;
                    c ^= helper::read_int(slice::from_raw_parts(ptr.offset(16), excessive));

                    // Diffuse.
                    a = helper::diffuse(a);
                    b = helper::diffuse(b);
                    c = helper::diffuse(c);
                },
                24 => {
                    // 24 bytes excessive.

                    // Mix in the partial block.
                    a ^= helper::read_u64(ptr);
                    b ^= helper::read_u64(ptr.offset(8));
                    c ^= helper::read_u64(ptr.offset(16));

                    // Diffuse.
                    a = helper::diffuse(a);
                    b = helper::diffuse(b);
                    c = helper::diffuse(c);
                },
                _ => {
                    // More than 24 bytes excessive.

                    // Mix in the partial block.
                    a ^= helper::read_u64(ptr);
                    b ^= helper::read_u64(ptr.offset(8));
                    c ^= helper::read_u64(ptr.offset(16));

                    // Write the last excessive bytes (<8 bytes).
                    excessive = excessive - 24;
                    d ^= helper::read_int(slice::from_raw_parts(ptr.offset(24), excessive));

                    // Diffuse.
                    a = helper::diffuse(a);
                    b = helper::diffuse(b);
                    c = helper::diffuse(c);
                    d = helper::diffuse(d);
                }
            }
        }

        SeaHash {
            a: a,
            b: b,
            c: c,
            d: d,
            written: written + buf.len() as u64,
        }
    }
    
    // Note: the following `hash_u*` / `hash_i*` functions could be removed
    // entirely (implementations moved to `Hashable::hash` impls). The only
    // reason I haven't done this is because micro-benchmarks show better
    // performance using these functions; not sure why.
    
    /// Hash a value to the given state.
    pub fn hash_u8(self, input: u8) -> Self {
        hash_num!(self, input as u64, 1)
    }
    
    /// Hash a value to the given state.
    pub fn hash_u16(self, input: u16) -> Self {
        hash_num!(self, input as u64, 2)
    }
    
    /// Hash a value to the given state.
    pub fn hash_u32(self, input: u32) -> Self {
        hash_num!(self, input as u64, 4)
    }
    
    /// Hash a value to the given state.
    pub fn hash_u64(self, input: u64) -> Self {
        hash_num!(self, input, 8)
    }
    
    /// Hash a value to the given state.
    pub fn hash_i8(self, input: i8) -> Self {
        hash_num!(self, input as u64, 1)
    }
    
    /// Hash a value to the given state.
    pub fn hash_i16(self, input: i16) -> Self {
        hash_num!(self, input as u64, 2)
    }
    
    /// Hash a value to the given state.
    pub fn hash_i32(self, input: i32) -> Self {
        hash_num!(self, input as u64, 4)
    }
    
    /// Hash a value to the given state.
    pub fn hash_i64(self, input: i64) -> Self {
        hash_num!(self, input as u64, 8)
    }
    
    /// Finalize the state.
    fn finalize(self) -> u64 {
        let SeaHash { written, mut a, b, mut c, d } = self;

        // XOR the states together. Even though XOR is commutative, it doesn't matter, because the
        // state vector's initial components are mutually distinct, and thus swapping even and odd
        // chunks will affect the result, because it is sensitive to the initial condition.
        a ^= b;
        c ^= d;
        a ^= c;
        // XOR the number of written bytes in order to make the excessive bytes zero-sensitive
        // (without this, two excessive zeros would be equivalent to three excessive zeros). This
        // is know as length padding.
        a ^= written;

        // We diffuse to make the excessive bytes discrete (i.e. small changes shouldn't give small
        // changes in the output).
        helper::diffuse(a)
    }
    
    /// Finalize the state, with 128 bits output (not crypto quality; see notes on type).
    fn finalize128(self) -> (u64, u64) {
        let SeaHash { written, mut a, mut b, c, d } = self;

        a ^= c;
        b ^= d;
        
        // As in finalize(), XOR with number of bytes written. We don't need this to affect all
        // parts of output. The final diffuse(a) is simply to diffuse small changes in `written`.
        a ^= written;
        (helper::diffuse(a), b)
    }
    
    /// Finalize the state, with 256 bits output (not crypto quality; see notes on type).
    fn finalize256(self) -> (u64, u64, u64, u64) {
        let SeaHash { written, mut a, b, c, d } = self;

        // As in finalize(), XOR with number of bytes written. We don't need this to affect all
        // parts of output. The final diffuse(a) is simply to diffuse small changes in `written`.
        a ^= written;
        (helper::diffuse(a), b, c, d)
    }
}

/// Hashable types
pub trait Hashable {
    fn hash(state: SeaHash, input: Self) -> SeaHash;
}

impl<'a> Hashable for &'a [u8] {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_buf(input)
    }
}
impl<'a> Hashable for &'a str {
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_buf(input.as_bytes())
    }
}
#[cfg(features="std")]
impl Hashable for String {
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_buf(input.as_bytes())
    }
}

impl Hashable for u8 {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_u8(input)
    }
}
impl Hashable for u16 {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_u16(input)
    }
}
impl Hashable for u32 {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_u32(input)
    }
}
impl Hashable for u64 {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_u64(input)
    }
}

impl Hashable for i8 {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_i8(input)
    }
}
impl Hashable for i16 {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_i16(input)
    }
}
impl Hashable for i32 {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_i32(input)
    }
}
impl Hashable for i64 {
    #[inline(always)]
    fn hash(state: SeaHash, input: Self) -> SeaHash {
        state.hash_i64(input)
    }
}

impl From<SeaHash> for [u8; 8] {
    #[inline] fn from(state: SeaHash) -> Self {
        let x = state.finalize().to_le();
        unsafe{ *(&x as *const u64 as *const [u8; 8]) }
    }
}

impl From<SeaHash> for [u8; 16] {
    #[inline] fn from(state: SeaHash) -> Self {
        let (a, b) = state.finalize128();
        let x = [a.to_le(), b.to_le()];
        unsafe{ *(&x as *const [u64; 2] as *const [u8; 16]) }
    }
}

impl From<SeaHash> for [u8; 32] {
    #[inline] fn from(state: SeaHash) -> Self {
        let (a, b, c, d) = state.finalize256();
        let x = [a.to_le(), b.to_le(), c.to_le(), d.to_le()];
        unsafe{ *(&x as *const [u64; 4] as *const [u8; 32]) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shakespear() {
        assert_eq!(SeaHash::hash_fixed("to be or not to be").finalize(),
                1988685042348123509);
        assert_eq!(SeaHash::hash_fixed("love is a wonderful terrible thing")
                .finalize(), 4784284276849692846);
    }
    
    #[test]
    fn fixed() {
        let text1 = "to be or not to be";
        let text2 = "To say the truth, reason and love keep little company together now-a-days.";
        let key = (1, 2, 3, 4);
        
        assert_eq!(SeaHash::new(key).hash(text1).finalize(),
            17668174308057396010);
        assert_eq!(SeaHash::new(key).hash(text1).finalize128(),
            (5668955536430251100, 580969811955034951));
        assert_eq!(SeaHash::new(key).hash(text1).finalize256(),
            (15539999726964083260, 580969811955034947, 12487322460008013292, 4));
        
        assert_eq!(SeaHash::new(key).hash(text2).finalize(),
            9612626706651696580);
        assert_eq!(SeaHash::new(key).hash(text2).finalize128(),
            (98617549140793774, 13535749936723083946));
        assert_eq!(SeaHash::new(key).hash(text2).finalize256(),
            (15714174257805948427, 7681793730966948527, 5582628158086256701, 15079056956347828229));
    }
    
    #[test]
    fn nums() {
        // Tests on some numbers. No external reference.
        assert_eq!(SeaHash::new_fixed().finalize(), 14492805990617963705);
        assert_eq!(SeaHash::hash_fixed(1).finalize(), 4319296126225194558);
        assert_eq!(SeaHash::hash_fixed(2).finalize(), 10988112147709237335);
        assert_eq!(SeaHash::hash_fixed(3).finalize(), 16369569951930432123);
        
        // Note: same value, different type. Length (in bytes) affects output:
        assert_eq!(SeaHash::hash_fixed(0xbeefu16).finalize(), 14882276537536738750);
        assert_eq!(SeaHash::hash_fixed(0xbeefu32).finalize(), 2367289199683255885);
        assert_eq!(SeaHash::hash_fixed(0xbeefi32).finalize(), 2367289199683255885);
        assert_eq!(SeaHash::hash_fixed(0xbeefu64).finalize(), 17783250324651728131);
        
        // Same value as above, in a byte slice (LE):
        assert_eq!(SeaHash::hash_fixed(&[0xefu8, 0xbe][..]).finalize(), 14882276537536738750);
        
        assert_eq!(SeaHash::hash_fixed(0x73510F2CE6203AA3i64).finalize(),
            SeaHash::hash_fixed(&[0xA3u8, 0x3A, 0x20, 0xE6, 0x2C, 0xF, 0x51, 0x73][..]).finalize());
    }
}

mod helper {
    //! Helper functions.

    /// Read a buffer smaller than 8 bytes into an integer in little-endian.
    ///
    /// This assumes that `buf.len() < 8`. If this is not satisfied, the behavior is unspecified.
    #[inline(always)]
    pub fn read_int(buf: &[u8]) -> u64 {
        // Because we want to make sure that it is register allocated, we fetch this into a variable.
        // It will likely make no difference anyway, though.
        let ptr = buf.as_ptr();

        unsafe {
            // Break it down to reads of integers with widths in total spanning the buffer. This minimizes
            // the number of reads
            match buf.len() {
                // u8.
                1 => *ptr as u64,
                // u16.
                2 => (*(ptr as *const u16)).to_le() as u64,
                // u16 + u8.
                3 => {
                    let a = (*(ptr as *const u16)).to_le() as u64;
                    let b = *ptr.offset(2) as u64;

                    a | (b << 16)
                },
                // u32.
                4 => (*(ptr as *const u32)).to_le() as u64,
                // u32 + u8.
                5 => {
                    let a = (*(ptr as *const u32)).to_le() as u64;
                    let b = *ptr.offset(4) as u64;

                    a | (b << 32)
                },
                // u32 + u16.
                6 => {
                    let a = (*(ptr as *const u32)).to_le() as u64;
                    let b = (*(ptr.offset(4) as *const u16)).to_le() as u64;

                    a | (b << 32)
                },
                // u32 + u16 + u8.
                7 => {
                    let a = (*(ptr as *const u32)).to_le() as u64;
                    let b = (*(ptr.offset(4) as *const u16)).to_le() as u64;
                    let c = *ptr.offset(6) as u64;

                    a | (b << 32) | (c << 48)
                },
                _ => 0,
            }
        }
    }

    /// Read a little-endian 64-bit integer from some buffer.
    #[inline(always)]
    pub unsafe fn read_u64(ptr: *const u8) -> u64 {
        (*(ptr as *const u64)).to_le()
    }

    /// The diffusion function.
    ///
    /// This is a bijective function emitting chaotic behavior. Such functions are used as building
    /// blocks for hash functions.
    pub fn diffuse(mut x: u64) -> u64 {
        // These are derived from the PCG RNG's round. Thanks to @Veedrac for proposing this. The basic
        // idea is that we use dynamic shifts, which are determined by the input itself. The shift is
        // chosen by the higher bits, which means that changing those flips the lower bits, which
        // scatters upwards because of the multiplication.

        x = x.wrapping_mul(0x6eed0e9da4d94a4f);
        let a = x >> 32;
        let b = x >> 60;
        x ^= a >> b;
        x = x.wrapping_mul(0x6eed0e9da4d94a4f);

        x
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn diffuse_test(x: u64, y: u64) {
            assert_eq!(diffuse(x), y);
        }

        #[test]
        fn read_int_() {
            assert_eq!(read_int(&[2, 3]), 770);
            assert_eq!(read_int(&[3, 2]), 515);
            assert_eq!(read_int(&[3, 2, 5]), 328195);
        }

        #[test]
        fn read_u64_() {
            unsafe {
                assert_eq!(read_u64([1, 0, 0, 0, 0, 0, 0, 0].as_ptr()), 1);
                assert_eq!(read_u64([2, 1, 0, 0, 0, 0, 0, 0].as_ptr()), 258);
            }
        }

        #[test]
        fn diffuse_test_vectors() {
            diffuse_test(94203824938, 17289265692384716055);
            diffuse_test(0xDEADBEEF, 12110756357096144265);
            diffuse_test(0, 0);
            diffuse_test(1, 15197155197312260123);
            diffuse_test(2, 1571904453004118546);
            diffuse_test(3, 16467633989910088880);
        }
    }
}
