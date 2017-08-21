// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A wrapper around any Read to treat it as an RNG.

use std::fmt::Debug;
use std::io::Read;
use std::mem;

use {CryptoRng, CryptoError};

/// An RNG that reads random bytes straight from a `Read`. This will
/// work best with an infinite reader, but this is not required.
///
/// # Panics
///
/// It will panic if it there is insufficient data to fulfill a request.
///
/// # Example
///
/// ```rust
/// use rand::{ReadRng, distributions};
///
/// let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let mut rng = ReadRng::new(&data[..]);
/// println!("{:x}", distributions::uniform::<u32, _>(&mut rng));
/// ```
#[derive(Debug)]
pub struct ReadRng<R: Debug> {
    reader: R
}

impl<R: Read + Debug> ReadRng<R> {
    /// Create a new `ReadRng` from a `Read`.
    pub fn new(r: R) -> ReadRng<R> {
        ReadRng {
            reader: r
        }
    }
}

impl<R: Read + Debug> CryptoRng<CryptoError> for ReadRng<R> {
    fn try_next_u32(&mut self) -> Result<u32, CryptoError> {
        // This is designed for speed: reading a LE integer on a LE
        // platform just involves blitting the bytes into the memory
        // of the u32, similarly for BE on BE; avoiding byteswapping.
        let mut buf = [0; 4];
        fill(&mut self.reader, &mut buf).map_err(|_| CryptoError)?;
        Ok(unsafe { *(buf.as_ptr() as *const u32) })
    }
    fn try_next_u64(&mut self) -> Result<u64, CryptoError> {
        // see above for explanation.
        let mut buf = [0; 8];
        fill(&mut self.reader, &mut buf).map_err(|_| CryptoError)?;
        Ok(unsafe { *(buf.as_ptr() as *const u64) })
    }
    fn fill_bytes(&mut self, v: &mut [u8]) -> Result<(), CryptoError> {
        if v.len() == 0 { return Ok(()) }
        fill(&mut self.reader, v).map_err(|_| CryptoError)
    }
}

fn fill(r: &mut Read, mut buf: &mut [u8]) -> Result<(), CryptoError> {
    while buf.len() > 0 {
        match r.read(buf)? {
            0 => return Err(CryptoError),
            n => buf = &mut mem::replace(&mut buf, &mut [])[n..],
        }
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::ReadRng;
    use CryptoRng;

    #[test]
    fn test_reader_rng_u64() {
        // transmute from the target to avoid endianness concerns.
        let v = vec![0u8, 0, 0, 0, 0, 0, 0, 1,
                     0  , 0, 0, 0, 0, 0, 0, 2,
                     0,   0, 0, 0, 0, 0, 0, 3];
        let mut rng = ReadRng::new(&v[..]);

        assert_eq!(rng.try_next_u64(), Ok(1_u64.to_be()));
        assert_eq!(rng.try_next_u64(), Ok(2_u64.to_be()));
        assert_eq!(rng.try_next_u64(), Ok(3_u64.to_be()));
    }
    #[test]
    fn test_reader_rng_u32() {
        let v = vec![0u8, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3];
        let mut rng = ReadRng::new(&v[..]);

        assert_eq!(rng.try_next_u32(), Ok(1_u32.to_be()));
        assert_eq!(rng.try_next_u32(), Ok(2_u32.to_be()));
        assert_eq!(rng.try_next_u32(), Ok(3_u32.to_be()));
    }
    #[test]
    fn test_reader_rng_fill_bytes() {
        let v = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut w = [0u8; 8];

        let mut rng = ReadRng::new(&v[..]);
        rng.fill_bytes(&mut w);

        assert!(v == w);
    }

    #[test]
    fn test_reader_rng_insufficient_bytes() {
        let mut rng = ReadRng::new(&[][..]);
        let mut v = [0u8; 3];
        assert!(rng.fill_bytes(&mut v).is_err());
    }
}
