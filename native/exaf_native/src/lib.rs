use rustler::{Env, Term};

mod array;
mod datatypes;

use array::*;
use datatypes::ExAfRef;

fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(ExAfRef, env);
    true
}

rustler::init!(
    "Elixir.ExAF.Native",
    [
        // Backend management
        backend_deallocate,
        // Conversion
        from_binary,
        to_binary,
        // Creation
        iota,
        // Elementwise - Exponentation
        exp,
        expm1,
        log,
        log1p,
        sigmoid,
        // Elementwise - Trignomentry
        sin,
        cos,
        tan,
        sinh,
        cosh,
        tanh,
        asin,
        acos,
        atan,
        asinh,
        acosh,
        atanh,
        // Elementwise - Error Functions
        erf,
        erfc,
        // Elementwise - Roots
        sqrt,
        rsqrt,
        cbrt,
        // Elementwise - Number Theoryesque Functions
        abs,
        floor,
        round,
        ceil,
        real,
        imag,
        // Shape
        reshape,
        // Type
        as_type
    ],
    load = load
);
