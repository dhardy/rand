// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Functions for randomly accessing and sampling sequences.
//! 
//! TODO: module doc

#[cfg(feature="alloc")] use core::ops::Index;
#[cfg(feature="alloc")] use core::slice;

#[cfg(feature="std")] use std::vec;
#[cfg(all(feature="alloc", not(feature="std")))] use alloc::{vec, Vec};
// BTreeMap is not as fast in tests, but better than nothing.
#[cfg(feature="std")] use std::collections::HashMap;
#[cfg(all(feature="alloc", not(feature="std")))] use alloc::btree_map::BTreeMap;


use super::Rng;

/// Extension trait on slices, providing random mutation and sampling methods.
/// 
/// An implementation is provided for slices. This may also be implementable for
/// other types.
pub trait SliceRandom {
    /// The element type.
    type Item;

    /// Returns a reference to one random element of the slice, or `None` if the
    /// slice is empty.
    /// 
    /// Depending on the implementation, complexity is expected to be `O(1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::thread_rng;
    /// use rand::seq::SliceRandom;
    ///
    /// let choices = [1, 2, 4, 8, 16, 32];
    /// let mut rng = thread_rng();
    /// println!("{:?}", choices.choose(&mut rng));
    /// assert_eq!(choices[..0].choose(&mut rng), None);
    /// ```
    fn choose<R>(&self, rng: &mut R) -> Option<&Self::Item>
        where R: Rng + ?Sized;

    /// Returns a mutable reference to one random element of the slice, or
    /// `None` if the slice is empty.
    /// 
    /// Depending on the implementation, complexity is expected to be `O(1)`.
    fn choose_mut<R>(&mut self, rng: &mut R) -> Option<&mut Self::Item>
        where R: Rng + ?Sized;

    /// Produces an iterator that chooses `amount` elements from the slice at
    /// random without repeating any.
    ///
    /// In case this API is not sufficiently flexible, use `sample_indices` then
    /// apply the indices to the slice.
    /// 
    /// If `shuffled == true` then the sampled values will be fully shuffled;
    /// otherwise the values may only partially shuffled, depending on the
    /// algorithm used (i.e. biases may exist in the ordering of sampled
    /// elements). Depending on the algorithm used internally, full shuffling
    /// may add significant overhead for `amount` > 10 or so, but not more
    /// than double the time and often much less.
    ///
    /// Complexity is expected to be the same as `sample_indices`.
    /// 
    /// # Example
    /// ```
    /// use rand::seq::SliceRandom;
    /// 
    /// let mut rng = &mut rand::thread_rng();
    /// let sample = "Hello, audience!".as_bytes();
    /// 
    /// // collect the results into a vector:
    /// let v: Vec<u8> = sample.choose_multiple(&mut rng, 3, true).cloned().collect();
    /// 
    /// // store in a buffer:
    /// let mut buf = [0u8; 5];
    /// for (b, slot) in sample.choose_multiple(&mut rng, buf.len(), true).zip(buf.iter_mut()) {
    ///     *slot = *b;
    /// }
    /// ```
    #[cfg(feature = "alloc")]
    fn choose_multiple<R>(&self, rng: &mut R, amount: usize, shuffled: bool) -> SliceChooseIter<Self, Self::Item>
        where R: Rng + ?Sized;

    /// Shuffle a mutable slice in place.
    /// 
    /// Depending on the implementation, complexity is expected to be `O(1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::thread_rng;
    /// use rand::seq::SliceRandom;
    ///
    /// let mut rng = thread_rng();
    /// let mut y = [1, 2, 3, 4, 5];
    /// println!("Unshuffled: {:?}", y);
    /// y.shuffle(&mut rng);
    /// println!("Shuffled:   {:?}", y);
    /// ```
    fn shuffle<R>(&mut self, rng: &mut R) where R: Rng + ?Sized;

    /// Shuffle a slice in place, but exit early.
    ///
    /// Returns two mutable slices from the source slice. The first contains
    /// `amount` elements randomly permuted. The second has the remaining
    /// elements that are not fully shuffled.
    ///
    /// This is an efficient method to select `amount` elements at random from
    /// the slice, provided the slice may be mutated.
    ///
    /// If you only need to choose elements randomly and `amount > self.len()/2`
    /// then you may improve performance by taking
    /// `amount = values.len() - amount` and using only the second slice.
    ///
    /// If `amount` is greater than the number of elements in the slice, this
    /// will perform a full shuffle.
    ///
    /// Complexity is expected to be `O(m)` where `m = amount`.
    fn partial_shuffle<R>(&mut self, rng: &mut R, amount: usize)
        -> (&mut [Self::Item], &mut [Self::Item]) where R: Rng + ?Sized;
}

/// Extension trait on iterators, providing random sampling methods.
pub trait IteratorRandom: Iterator + Sized {
    /// Choose one element at random from the iterator. If you have a slice,
    /// it's significantly faster to call the [`choose`] or [`choose_mut`]
    /// functions using the slice instead.
    ///
    /// Returns `None` if and only if the iterator is empty.
    /// 
    /// Complexity is `O(n)`, where `n` is the length of the iterator.
    /// This likely consumes multiple random numbers, but the exact number
    /// is unspecified.
    /// [`choose`]: trait.SliceRandom.html#method.choose
    /// [`choose_mut`]: trait.SliceRandom.html#method.choose_mut
    fn choose<R>(mut self, rng: &mut R) -> Option<Self::Item>
        where R: Rng + ?Sized
    {
        if let Some(elem) = self.next() {
            let mut result = elem;
            
            // Continue until the iterator is exhausted
            for (i, elem) in self.enumerate() {
                let denom = (i + 2) as f64; // accurate to 2^53 elements
                if rng.gen_bool(1.0 / denom) {
                    result = elem;
                }
            }
            
            Some(result)
        } else {
            None
        }
    }

    /// Collects `amount` values at random from the iterator into a supplied
    /// buffer.
    /// 
    /// Although the elements are selected randomly, the order of elements in
    /// the buffer is neither stable nor fully random. If random ordering is
    /// desired, shuffle the result.
    /// 
    /// Returns the number of elements added to the buffer. This equals `amount`
    /// unless the iterator contains insufficient elements, in which case this
    /// equals the number of elements available.
    /// 
    /// Complexity is `O(n)` where `n` is the length of the iterator.
    fn choose_multiple_fill<R>(mut self, rng: &mut R, buf: &mut [Self::Item])
        -> usize where R: Rng + ?Sized
    {
        let amount = buf.len();
        let mut len = 0;
        while len < amount {
            if let Some(elem) = self.next() {
                buf[len] = elem;
                len += 1;
            } else {
                // Iterator exhausted; stop early
                return len;
            }
        }

        // Continue, since the iterator was not exhausted
        for (i, elem) in self.enumerate() {
            let k = rng.gen_range(0, i + 1 + amount);
            if let Some(slot) = buf.get_mut(k) {
                *slot = elem;
            }
        }
        len
    }

    /// Collects `amount` values at random from the iterator into a vector.
    ///
    /// This is equivalent to `choose_multiple_fill` except for the result type.
    ///
    /// Although the elements are selected randomly, the order of elements in
    /// the buffer is neither stable nor fully random. If random ordering is
    /// desired, shuffle the result.
    /// 
    /// The length of the returned vector equals `amount` unless the iterator
    /// contains insufficient elements, in which case it equals the number of
    /// elements available.
    /// 
    /// Complexity is `O(n)` where `n` is the length of the iterator.
    #[cfg(feature = "alloc")]
    fn choose_multiple<R>(mut self, rng: &mut R, amount: usize) -> Vec<Self::Item>
        where R: Rng + ?Sized
    {
        let mut reservoir = Vec::with_capacity(amount);
        reservoir.extend(self.by_ref().take(amount));

        // Continue unless the iterator was exhausted
        //
        // note: this prevents iterators that "restart" from causing problems.
        // If the iterator stops once, then so do we.
        if reservoir.len() == amount {
            for (i, elem) in self.enumerate() {
                let k = rng.gen_range(0, i + 1 + amount);
                if let Some(slot) = reservoir.get_mut(k) {
                    *slot = elem;
                }
            }
        } else {
            // Don't hang onto extra memory. There is a corner case where
            // `amount` was much less than `self.len()`.
            reservoir.shrink_to_fit();
        }
        reservoir
    }
}


impl<T> SliceRandom for [T] {
    type Item = T;

    fn choose<R>(&self, rng: &mut R) -> Option<&Self::Item>
        where R: Rng + ?Sized
    {
        if self.is_empty() {
            None
        } else {
            Some(&self[rng.gen_range(0, self.len())])
        }
    }

    fn choose_mut<R>(&mut self, rng: &mut R) -> Option<&mut Self::Item>
        where R: Rng + ?Sized
    {
        if self.is_empty() {
            None
        } else {
            let len = self.len();
            Some(&mut self[rng.gen_range(0, len)])
        }
    }

    #[cfg(feature = "alloc")]
    fn choose_multiple<R>(&self, rng: &mut R, amount: usize, shuffled: bool)
        -> SliceChooseIter<Self, Self::Item>
        where R: Rng + ?Sized
    {
        let amount = ::core::cmp::min(amount, self.len());
        SliceChooseIter {
            slice: self,
            _phantom: Default::default(),
            indices: sample_indices(rng, self.len(), amount, shuffled).into_iter_usize(),
        }
    }

    fn shuffle<R>(&mut self, rng: &mut R) where R: Rng + ?Sized
    {
        for i in (1..self.len()).rev() {
            // invariant: elements with index > i have been locked in place.
            self.swap(i, rng.gen_range(0, i + 1));
        }
    }

    fn partial_shuffle<R>(&mut self, rng: &mut R, amount: usize)
        -> (&mut [Self::Item], &mut [Self::Item]) where R: Rng + ?Sized
    {
        // This applies Durstenfeld's algorithm for the
        // [Fisher–Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm)
        // for an unbiased permutation, but exits early after choosing `amount`
        // elements.
        
        let len = self.len();
        let end = if amount >= len { 0 } else { len - amount };
        
        for i in (end..len).rev() {
            // invariant: elements with index > i have been locked in place.
            self.swap(i, rng.gen_range(0, i + 1));
        }
        let r = self.split_at_mut(end);
        (r.1, r.0)
    }
}

impl<I> IteratorRandom for I where I: Iterator + Sized {}


/// Iterator over multiple choices, as returned by [`SliceRandom::choose_multiple](
/// trait.SliceRandom.html#method.choose_multiple).
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct SliceChooseIter<'a, S: ?Sized + 'a, T: 'a> {
    slice: &'a S,
    _phantom: ::core::marker::PhantomData<T>,
    indices: IndicesIntoIter,
}

#[cfg(feature = "alloc")]
impl<'a, S: Index<usize, Output = T> + ?Sized + 'a, T: 'a> Iterator for SliceChooseIter<'a, S, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: investigate using SliceIndex::get_unchecked when stable
        self.indices.next().map(|i| &(*self.slice)[i as usize])
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.indices.len(), Some(self.indices.len()))
    }
}

#[cfg(feature = "alloc")]
impl<'a, S: Index<usize, Output = T> + ?Sized + 'a, T: 'a> ExactSizeIterator
    for SliceChooseIter<'a, S, T>
{
    fn len(&self) -> usize {
        self.indices.len()
    }
}


/// Randomly sample `amount` elements from a finite iterator.
///
/// Deprecated: use [`IteratorRandom::choose_multiple`] instead.
/// 
/// [`IteratorRandom::choose_multiple`]: trait.IteratorRandom.html#method.choose_multiple
#[cfg(feature = "alloc")]
#[deprecated(since="0.6.0", note="use IteratorRandom::choose_multiple instead")]
pub fn sample_iter<T, I, R>(rng: &mut R, iterable: I, amount: usize) -> Result<Vec<T>, Vec<T>>
    where I: IntoIterator<Item=T>,
          R: Rng + ?Sized,
{
    use seq::IteratorRandom;
    let iter = iterable.into_iter();
    let result = iter.choose_multiple(rng, amount);
    if result.len() == amount {
        Ok(result)
    } else {
        Err(result)
    }
}

/// Randomly sample exactly `amount` values from `slice`.
///
/// The values are non-repeating and in random order.
///
/// This implementation uses `O(amount)` time and memory.
///
/// Panics if `amount > slice.len()`
///
/// Deprecated: use [`SliceRandom::choose_multiple`] instead.
/// 
/// [`SliceRandom::choose_multiple`]: trait.SliceRandom.html#method.choose_multiple
#[cfg(feature = "alloc")]
#[deprecated(since="0.6.0", note="use SliceRandom::choose_multiple instead")]
pub fn sample_slice<R, T>(rng: &mut R, slice: &[T], amount: usize) -> Vec<T>
    where R: Rng + ?Sized,
          T: Clone
{
    let indices = sample_indices(rng, slice.len(), amount, true);

    let mut out = Vec::with_capacity(amount);
    out.extend(indices.iter_usize().map(|i| slice[i].clone()));
    out
}

/// Randomly sample exactly `amount` references from `slice`.
///
/// The references are non-repeating and in random order.
///
/// This implementation uses `O(amount)` time and memory.
///
/// Panics if `amount > slice.len()`
///
/// Deprecated: use [`SliceRandom::choose_multiple`] instead.
/// 
/// [`SliceRandom::choose_multiple`]: trait.SliceRandom.html#method.choose_multiple
#[cfg(feature = "alloc")]
#[deprecated(since="0.6.0", note="use SliceRandom::choose_multiple instead")]
pub fn sample_slice_ref<'a, R, T>(rng: &mut R, slice: &'a [T], amount: usize) -> Vec<&'a T>
    where R: Rng + ?Sized
{
    let indices = sample_indices(rng, slice.len(), amount, true);

    let mut out = Vec::with_capacity(amount);
    out.extend(indices.iter_usize().map(|i| &slice[i]));
    out
}

/// Return type of `sample_indices`.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub enum Indices {
    /// Representation: a vector over `u32` values
    U32(Vec<u32>)
}

#[cfg(feature = "alloc")]
impl Indices {
    /// Returns the number of indices
    pub fn len(&self) -> usize {
        match self {
            &Indices::U32(ref v) => v.len(),
        }
    }

    /// Return result as a `Vec<usize>`. Conversion may or may not be trivial.
    pub fn into_vec_usize(self) -> Vec<usize> {
        match self {
            Indices::U32(v) => v.into_iter().map(|i| i as usize).collect(),
        }
    }

    /// Iterate over the indices as a sequence of `usize` values
    pub fn iter_usize<'a>(&'a self) -> IndicesIter<'a> {
        match self {
            &Indices::U32(ref v) => IndicesIter::U32(v.iter()),
        }
    }
    
    /// Convert into an iterator over the indices as a sequence of `usize` values
    pub fn into_iter_usize(self) -> IndicesIntoIter {
        match self {
            Indices::U32(v) => IndicesIntoIter::U32(v.into_iter()),
        }
    }
}

#[cfg(feature = "alloc")]
impl From<Vec<u32>> for Indices {
    fn from(v: Vec<u32>) -> Self {
        Indices::U32(v)
    }
}

/// Return type of `Indices::iter_usize`.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub enum IndicesIter<'a> {
    #[doc(hidden)] U32(slice::Iter<'a, u32>),
}

#[cfg(feature = "alloc")]
impl<'a> Iterator for IndicesIter<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        use self::IndicesIter::*;
        match self {
            &mut U32(ref mut iter) => iter.next().map(|i| *i as usize),
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            &IndicesIter::U32(ref v) => v.size_hint(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<'a> ExactSizeIterator for IndicesIter<'a> {
    fn len(&self) -> usize {
        match self {
            &IndicesIter::U32(ref v) => v.len(),
        }
    }
}

/// Return type of `Indices::into_iter_usize`.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub enum IndicesIntoIter {
    #[doc(hidden)] U32(vec::IntoIter<u32>),
}

#[cfg(feature = "alloc")]
impl Iterator for IndicesIntoIter {
    type Item = usize;
    
    fn next(&mut self) -> Option<Self::Item> {
        use self::IndicesIntoIter::*;
        match self {
            &mut U32(ref mut v) => v.next().map(|i| i as usize),
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        use self::IndicesIntoIter::*;
        match self {
            &U32(ref v) => v.size_hint(),
        }
    }
}

#[cfg(feature = "alloc")]
impl ExactSizeIterator for IndicesIntoIter {
    fn len(&self) -> usize {
        use self::IndicesIntoIter::*;
        match self {
            &U32(ref v) => v.len(),
        }
    }
}


/// Randomly sample exactly `amount` distinct indices from `0..length`.
///
/// If `shuffled == true` then the sampled values will be fully shuffled;
/// otherwise the values may only partially shuffled, depending on the
/// algorithm used (i.e. biases may exist in the ordering of sampled elements).
/// Depending on the algorithm used internally, full shuffling may add
/// significant overhead for `amount` > 10 or so, but not more than double
/// the time and often much less.
///
/// This method is used internally by the slice sampling methods, but it can
/// sometimes be useful to have the indices themselves so this is provided as
/// an alternative.
///
/// The implementation used is not specified; we automatically select the
/// fastest available implementation for the `length` and `amount` parameters
/// (based on detailed profiling on an Intel Haswell CPU). Roughly speaking,
/// complexity is `O(amount)`, except that when `amount` is small, performance
/// is closer to `O(amount^2)`, and when `length` is close to `amount` then
/// `O(length)`.
///
/// Note that we only support `u32` indices since this covers the vast majority
/// of uses, and performance is significantly better than with `u64`.
/// 
/// If an allocation-free `no_std` function is required, it is suggested
/// to adapt the internal `sample_indices_floyd` implementation.
///
/// Panics if `amount > length` or if `length` is not reprentable as a `u32`.
#[cfg(feature = "alloc")]
pub fn sample_indices<R>(rng: &mut R, length: usize, amount: usize,
    shuffled: bool) -> Indices
    where R: Rng + ?Sized,
{
    if amount > length {
        panic!("`amount` of samples must be less than or equal to `length`");
    }
    if length > (::core::u32::MAX as usize) {
        panic!("`length` is not representable as `u32`");
    }
    let amount = amount as u32;
    let length = length as u32;
    
    // Choice of algorithm here depends on both length and amount. See:
    // https://github.com/rust-lang-nursery/rand/pull/479
    // We do some calculations with f32. Accuracy is not very important.

    if amount < 442 {
        const C: [[f32; 2]; 2] = [[1.2, 6.0/45.0], [10.0, 70.0/9.0]];
        let j = if length < 500_000 { 0 } else { 1 };
        let amount_fp = amount as f32;
        let m4 = C[0][j] * amount_fp;
        // Short-cut: when amount < 12, floyd's is always faster
        if amount > 11 && (length as f32) < (C[1][j] + m4) * amount_fp {
            sample_indices_inplace(rng, length, amount)
        } else {
            sample_indices_floyd(rng, length, amount, shuffled)
        }
    } else {
        const C: [f32; 2] = [590.0, 600.0/9.0];
        let j = if length < 500_000 { 0 } else { 1 };
        if (length as f32) < C[j] * (amount as f32) {
            sample_indices_inplace(rng, length, amount)
        } else {
            sample_indices_cache(rng, length, amount)
        }
    }
}

/// Randomly sample exactly `amount` indices from `0..length`, using Floyd's
/// combination algorithm.
/// 
/// If `shuffled == false`, the values are only partially shuffled (i.e. biases
/// exist in the ordering of sampled elements). If `shuffled == true`, the
/// values are fully shuffled.
///
/// This implementation uses `O(amount)` memory and `O(amount^2)` time.
#[cfg(feature = "alloc")]
fn sample_indices_floyd<R>(rng: &mut R, length: u32, amount: u32, shuffled: bool) -> Indices
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut indices = Vec::with_capacity(amount as usize);
    for j in length - amount .. length {
        let t = rng.gen_range(0, j + 1);
        if indices.contains(&t) {
            indices.push(j)
        } else {
            indices.push(t)
        };
    }
    if shuffled {
        // Note that there is a variant of Floyd's algorithm with native full
        // shuffling, but it is slow because it requires arbitrary insertions.
        indices.shuffle(rng);
    }
    Indices::from(indices)
}

/// Randomly sample exactly `amount` indices from `0..length`, using an inplace
/// partial Fisher-Yates method.
/// Sample an amount of indices using an inplace partial fisher yates method.
///
/// This allocates the entire `length` of indices and randomizes only the first `amount`.
/// It then truncates to `amount` and returns.
/// 
/// This method is not appropriate for large `length` and potentially uses a lot
/// of memory; because of this we only implement for `u32` index (which improves
/// performance in all cases).
///
/// This is likely the fastest for small lengths since it avoids the need for
/// allocations. Set-up is `O(length)` time and memory and shuffling is
/// `O(amount)` time.
#[cfg(feature = "alloc")]
fn sample_indices_inplace<R>(rng: &mut R, length: u32, amount: u32) -> Indices
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    let mut indices: Vec<u32> = Vec::with_capacity(length as usize);
    indices.extend(0..length);
    for i in 0..amount {
        let j: u32 = rng.gen_range(i, length);
        indices.swap(i as usize, j as usize);
    }
    indices.truncate(amount as usize);
    debug_assert_eq!(indices.len(), amount as usize);
    Indices::from(indices)
}

/// Randomly sample exactly `amount` indices from `0..length`, using a
/// dynamically-cached partial Fisher-Yates method.
///
/// The cache avoids allocating the entire `length` of values. This is
/// especially useful when `amount <<< length`; e.g. selecting 3 non-repeating
/// values from `1_000_000`. The algorithm is `O(amount)` time and memory,
/// but due to overheads will often be slower than other approaches.
#[cfg(feature = "alloc")]
fn sample_indices_cache<R>(rng: &mut R, length: u32, amount: u32) -> Indices
    where R: Rng + ?Sized,
{
    debug_assert!(amount <= length);
    #[cfg(feature="std")] let mut cache = HashMap::with_capacity(amount as usize);
    #[cfg(not(feature="std"))] let mut cache = BTreeMap::new();
    let mut indices = Vec::with_capacity(amount as usize);
    for i in 0..amount {
        let j: u32 = rng.gen_range(i, length);

        // get the current values at i and j ...
        let x_i = match cache.get(&i) {
            Some(x) => *x,
            None => i,
        };
        let x_j = match cache.get(&j) {
            Some(x) => *x,
            None => j,
        };

        // ... and swap them
        cache.insert(j, x_i);
        indices.push(x_j);  // push at position i
    }
    debug_assert_eq!(indices.len(), amount as usize);
    Indices::from(indices)
}

#[cfg(test)]
mod test {
    use super::*;
    #[cfg(feature = "alloc")] use {Rng, SeedableRng};
    #[cfg(feature = "alloc")] use prng::XorShiftRng;
    #[cfg(all(feature="alloc", not(feature="std")))]
    use alloc::Vec;

    #[test]
    fn test_slice_choose() {
        let mut r = ::test::rng(107);
        let chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'];
        let mut chosen = [0i32; 14];
        for _ in 0..1000 {
            let picked = *chars.choose(&mut r).unwrap();
            chosen[(picked as usize) - ('a' as usize)] += 1;
        }
        for count in chosen.iter() {
            let err = *count - (1000 / (chars.len() as i32));
            assert!(-20 <= err && err <= 20);
        }

        chosen.iter_mut().for_each(|x| *x = 0);
        for _ in 0..1000 {
            *chosen.choose_mut(&mut r).unwrap() += 1;
        }
        for count in chosen.iter() {
            let err = *count - (1000 / (chosen.len() as i32));
            assert!(-20 <= err && err <= 20);
        }

        let mut v: [isize; 0] = [];
        assert_eq!(v.choose(&mut r), None);
        assert_eq!(v.choose_mut(&mut r), None);
    }

    #[test]
    fn test_iterator_choose() {
        let mut r = ::test::rng(109);
        let mut chosen = [0i32; 9];
        for _ in 0..1000 {
            let picked = (0..9).choose(&mut r).unwrap();
            chosen[picked] += 1;
        }
        for count in chosen.iter() {
            let err = *count - 1000 / 9;
            assert!(-25 <= err && err <= 25);
        }

        assert_eq!((0..0).choose(&mut r), None);
    }

    #[test]
    fn test_shuffle() {

        let mut r = ::test::rng(108);
        let empty: &mut [isize] = &mut [];
        empty.shuffle(&mut r);
        let mut one = [1];
        one.shuffle(&mut r);
        let b: &[_] = &[1];
        assert_eq!(one, b);

        let mut two = [1, 2];
        two.shuffle(&mut r);
        assert!(two == [1, 2] || two == [2, 1]);

        fn move_last(slice: &mut [usize], pos: usize) {
            // use slice[pos..].rotate_left(1); once we can use that
            let last_val = slice[pos];
            for i in pos..slice.len() - 1 {
                slice[i] = slice[i + 1];
            }
            *slice.last_mut().unwrap() = last_val;
        }
        let mut counts = [0i32; 24];
        for _ in 0..10000 {
            let mut arr: [usize; 4] = [0, 1, 2, 3];
            arr.shuffle(&mut r);
            let mut permutation = 0usize;
            let mut pos_value = counts.len();
            for i in 0..4 {
                pos_value /= 4 - i;
                let pos = arr.iter().position(|&x| x == i).unwrap();
                assert!(pos < (4 - i));
                permutation += pos * pos_value;
                move_last(&mut arr, pos);
                assert_eq!(arr[3], i);
            }
            for i in 0..4 {
                assert_eq!(arr[i], i);
            }
            counts[permutation] += 1;
        }
        for count in counts.iter() {
            let err = *count - 10000i32 / 24;
            assert!(-50 <= err && err <= 50);
        }
    }
    
    #[test]
    fn test_partial_shuffle() {
        let mut r = ::test::rng(118);
        
        let mut empty: [u32; 0] = [];
        let res = empty.partial_shuffle(&mut r, 10);
        assert_eq!((res.0.len(), res.1.len()), (0, 0));
        
        let mut v = [1, 2, 3, 4, 5];
        let res = v.partial_shuffle(&mut r, 2);
        assert_eq!((res.0.len(), res.1.len()), (2, 3));
        assert!(res.0[0] != res.0[1]);
        // First elements are only modified if selected, so at least one isn't modified:
        assert!(res.1[0] == 1 || res.1[1] == 2 || res.1[2] == 3);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn test_sample_iter() {
        let min_val = 1;
        let max_val = 100;

        let mut r = ::test::rng(401);
        let vals = (min_val..max_val).collect::<Vec<i32>>();
        let small_sample = vals.iter().choose_multiple(&mut r, 5);
        let large_sample = vals.iter().choose_multiple(&mut r, vals.len() + 5);

        assert_eq!(small_sample.len(), 5);
        assert_eq!(large_sample.len(), vals.len());
        // no randomization happens when amount >= len
        assert_eq!(large_sample, vals.iter().collect::<Vec<_>>());

        assert!(small_sample.iter().all(|e| {
            **e >= min_val && **e <= max_val
        }));
    }
    
    #[test]
    #[cfg(feature = "alloc")]
    #[allow(deprecated)]
    fn test_sample_slice_boundaries() {
        let empty: &[u8] = &[];

        let mut r = ::test::rng(402);

        // sample 0 items
        assert_eq!(&sample_slice(&mut r, empty, 0)[..], [0u8; 0]);
        assert_eq!(&sample_slice(&mut r, &[42, 2, 42], 0)[..], [0u8; 0]);

        // sample 1 item
        assert_eq!(&sample_slice(&mut r, &[42], 1)[..], [42]);
        let v = sample_slice(&mut r, &[1, 42], 1)[0];
        assert!(v == 1 || v == 42);

        // sample "all" the items
        let v = sample_slice(&mut r, &[42, 133], 2);
        assert!(&v[..] == [42, 133] || v[..] == [133, 42]);

        assert_eq!(sample_indices_inplace(&mut r, 0, 0).len(), 0);
        assert_eq!(sample_indices_inplace(&mut r, 1, 0).len(), 0);
        assert_eq!(sample_indices_inplace(&mut r, 1, 1).into_vec_usize(), vec![0]);

        assert_eq!(sample_indices_cache(&mut r, 0, 0).len(), 0);
        assert_eq!(sample_indices_cache(&mut r, 1, 0).len(), 0);
        assert_eq!(sample_indices_cache(&mut r, 1, 1).into_vec_usize(), vec![0]);

        assert_eq!(sample_indices_floyd(&mut r, 0, 0, false).len(), 0);
        assert_eq!(sample_indices_floyd(&mut r, 1, 0, false).len(), 0);
        assert_eq!(sample_indices_floyd(&mut r, 1, 1, false).into_vec_usize(), vec![0]);
        
        // These algorithms should be fast with big numbers. Test average.
        let indices = sample_indices_cache(&mut r, 1 << 25, 10);
        let sum: usize = indices.iter_usize().sum();
        assert!(1 << 25 < sum && sum < (1 << 25) * 25);
        
        let indices = sample_indices_floyd(&mut r, 1 << 25, 10, false);
        let sum: usize = indices.iter_usize().sum();
        assert!(1 << 25 < sum && sum < (1 << 25) * 25);

        // Make sure lucky 777's aren't lucky
        let slice = &[42, 777];
        let mut num_42 = 0;
        let total = 1000;
        for _ in 0..total {
            let v = sample_slice(&mut r, slice, 1);
            assert_eq!(v.len(), 1);
            let v = v[0];
            assert!(v == 42 || v == 777);
            if v == 42 {
                num_42 += 1;
            }
        }
        let ratio_42 = num_42 as f64 / 1000 as f64;
        assert!(0.4 <= ratio_42 || ratio_42 <= 0.6, "{}", ratio_42);
    }

    #[test]
    #[cfg(feature = "alloc")]
    #[allow(deprecated)]
    fn test_sample_slice() {
        let xor_rng = XorShiftRng::from_seed;

        let mut r = ::test::rng(403);

        for n in 1..20 {
            let length = 5*n - 4;   // 1, 6, ...
            let amount = r.gen_range(0, length);
            let mut seed = [0u8; 16];
            r.fill(&mut seed);

            // assert the basics work
            let regular = sample_indices(
                &mut xor_rng(seed), length, amount, true);
            assert_eq!(regular.len(), amount);
            assert!(regular.iter_usize().all(|e| e < length));

            // also test that sampling the slice works
            let vec: Vec<u32> = (0..(length as u32)).collect();
            let result = sample_slice(&mut xor_rng(seed), &vec, amount);
            assert_eq!(result, regular.iter_usize().map(|i| i as u32).collect::<Vec<_>>());

            let result = sample_slice_ref(&mut xor_rng(seed), &vec, amount);
            assert!(result.iter().zip(regular.iter_usize()).all(|(i,j)| **i == j as u32));
        }
    }
    
    #[test]
    #[cfg(feature = "alloc")]
    fn test_sample_alg() {
        let xor_rng = XorShiftRng::from_seed;

        let mut r = ::test::rng(403);
        let mut seed = [0u8; 16];
        
        // We can't test which algorithm is used directly, but Floyd's alg
        // should produce different results from the others. (Also, `inplace`
        // and `cached` currently use different sizes thus produce different results.)
        
        // A small length and relatively large amount should use inplace
        r.fill(&mut seed);
        let (length, amount): (usize, usize) = (100, 50);
        let v1 = sample_indices(&mut xor_rng(seed), length, amount, true);
        let v2 = sample_indices_inplace(&mut xor_rng(seed), length as u32, amount as u32);
        assert!(v1.iter_usize().all(|e| e < length));
        assert_eq!(v1, v2);
        
        // Test Floyd's alg does produce different results
        let v3 = sample_indices_floyd(&mut xor_rng(seed), length as u32, amount as u32, true);
        assert!(v1 != v3);
        
        // A large length and small amount should use Floyd
        r.fill(&mut seed);
        let (length, amount): (usize, usize) = (1<<20, 50);
        let v1 = sample_indices(&mut xor_rng(seed), length, amount, true);
        let v2 = sample_indices_floyd(&mut xor_rng(seed), length as u32, amount as u32, true);
        assert!(v1.iter_usize().all(|e| e < length));
        assert_eq!(v1, v2);
        
        // A large length and larger amount should use cache
        r.fill(&mut seed);
        let (length, amount): (usize, usize) = (1<<20, 600);
        let v1 = sample_indices(&mut xor_rng(seed), length, amount, true);
        let v2 = sample_indices_cache(&mut xor_rng(seed), length as u32, amount as u32);
        assert!(v1.iter_usize().all(|e| e < length));
        assert_eq!(v1, v2);
    }
}
