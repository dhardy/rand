// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unsafe debugging functions for inspecting values.

#[allow(missing_doc)];

use core::cast::transmute;
#[cfg(stage0)]
use intrinsic::{get_tydesc};
#[cfg(not(stage0))]
use core::unstable::intrinsics::{get_tydesc};

pub mod rustrt {
    #[cfg(stage0)]
    use intrinsic::{TyDesc};
    #[cfg(not(stage0))]
    use core::unstable::intrinsics::{TyDesc};

    #[abi = "cdecl"]
    pub extern {
        pub unsafe fn debug_tydesc(td: *TyDesc);
        pub unsafe fn debug_opaque(td: *TyDesc, x: *());
        pub unsafe fn debug_box(td: *TyDesc, x: *());
        pub unsafe fn debug_tag(td: *TyDesc, x: *());
        pub unsafe fn debug_fn(td: *TyDesc, x: *());
        pub unsafe fn debug_ptrcast(td: *TyDesc, x: *()) -> *();
        pub unsafe fn rust_dbg_breakpoint();
    }
}

pub fn debug_tydesc<T>() {
    unsafe {
        rustrt::debug_tydesc(get_tydesc::<T>());
    }
}

pub fn debug_opaque<T>(x: T) {
    unsafe {
        rustrt::debug_opaque(get_tydesc::<T>(), transmute(&x));
    }
}

pub fn debug_box<T>(x: @T) {
    unsafe {
        rustrt::debug_box(get_tydesc::<T>(), transmute(&x));
    }
}

pub fn debug_tag<T>(x: T) {
    unsafe {
        rustrt::debug_tag(get_tydesc::<T>(), transmute(&x));
    }
}

pub fn debug_fn<T>(x: T) {
    unsafe {
        rustrt::debug_fn(get_tydesc::<T>(), transmute(&x));
    }
}

pub unsafe fn ptr_cast<T, U>(x: @T) -> @U {
    transmute(
        rustrt::debug_ptrcast(get_tydesc::<T>(), transmute(x)))
}

/// Triggers a debugger breakpoint
pub fn breakpoint() {
    unsafe {
        rustrt::rust_dbg_breakpoint();
    }
}

#[test]
fn test_breakpoint_should_not_abort_process_when_not_under_gdb() {
    // Triggering a breakpoint involves raising SIGTRAP, which terminates
    // the process under normal circumstances
    breakpoint();
}
