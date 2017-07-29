rand
====

A Rust library for random number generators and other randomness functionality.

[![Build Status](https://travis-ci.org/rust-lang-nursery/rand.svg?branch=master)](https://travis-ci.org/rust-lang-nursery/rand)
[![Build status](https://ci.appveyor.com/api/projects/status/rm5c9o33k3jhchbw?svg=true)](https://ci.appveyor.com/project/alexcrichton/rand)

[Documentation](https://docs.rs/rand)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
rand = "0.3"
```

and this to your crate root:

```rust
extern crate rand;
```

## Examples

There is built-in support for a random number generator (RNG) associated with
each thread stored in thread-local storage. This RNG can be accessed via
thread_rng.

```rust
use rand::thread_rng;

let x: u32 = thread_rng().next_u32();
println!("{}", u32)
```

```rust
use rand::dist::{uniform};
use rand::Rng;

let mut rng = rand::thread_rng();
if uniform(&mut rng) { // random bool
    let x: i32 = uniform(&mut rng);
    let y: u32 = uniform(&mut rng);
    println!("i32: {}, u32: {}", x, y);
}
```

It is also possible to use other generators types, which have a similar interface. The following uses the "ChaCha" algorithm instead of the default.

```rust
use rand::{Rng, ChaChaRng};

let mut rng = rand::ChaChaRng::new_from_rng(&mut thread_rng());
println!("random between 0-9: {}", Range::new(0, 10).sample(&mut rng));
```


# License

`rand` is primarily distributed under the terms of both the MIT
license and the Apache License (Version 2.0), with portions covered by various
BSD-like licenses.

See LICENSE-APACHE, and LICENSE-MIT for details.
