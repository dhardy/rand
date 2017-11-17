initSidebarItems({"enum":[["ErrorKind","Error kind which can be matched over."]],"fn":[["random","Generates a random value using the thread-local random number generator."],["random_with","Generates a random value using the thread-local random number generator."],["thread_rng","Retrieve the lazily-initialized thread-local random number generator, seeded by the system. This is used by `random` and `random_with` to generate new values, and may be used directly with other distributions: `Range::new(0, 10).sample(&mut thread_rng())`."]],"mod":[["distributions","Sampling from random distributions."],["iter","Iterators attached to an `Rng`"],["jitter_rng","Non-physical true random number generator based on timing jitter."],["mock","Mock RNG implementations"],["prng","Pseudo random number generators are algorithms to produce apparently random numbers deterministically, and usually fairly quickly."],["reseeding","A wrapper around another RNG that reseeds it after it generates a certain number of random bytes."],["sequences","Random operations on sequences"],["utils","A collection of utility functions"]],"struct":[["Default","A generic random value distribution. Generates values using what appears to be \"the best\" distribution for each type, but ultimately the choice is arbitrary."],["Error","Error type of random number generators"],["OsRng","A random number generator that retrieves randomness straight from the operating system."],["ReadRng","An RNG that reads random bytes straight from a `Read`. This will work best with an infinite reader, but this is not required."],["StdRng","The standard RNG. This is designed to be efficient on the current platform."],["ThreadRng","The thread-local RNG."]],"trait":[["CryptoRng","A marker trait for an `Rng` which may be considered for use in cryptography."],["NewSeeded","Support mechanism for creating securely seeded objects  using the OS generator. Intended for use by RNGs, but not restricted to these."],["Rng","A random number generator (not necessarily suitable for cryptography)."],["Sample","Extension trait on [`Rng`] with some convenience methods."],["SeedFromRng","Support mechanism for creating random number generators seeded by other generators. All PRNGs should support this to enable `NewSeeded` support, which should be the preferred way of creating randomly-seeded generators."],["SeedableRng","A random number generator that can be explicitly seeded to produce the same stream of randomness multiple times."]]});