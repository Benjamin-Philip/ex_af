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
        eye,
        iota,
        // Elementwise - Arithmetic
        add,
        subtract,
        multiply,
        power,
        remainder,
        divide,
        min,
        max,
        // Elementwise - Comparison
        equal,
        not_equal,
        greater,
        less,
        greater_equal,
        less_equal,
        // Elementwise - Exponentation
        exp,
        expm1,
        log,
        log1p,
        sigmoid,
        // Elementwise - logical
        logical_and,
        logical_or,
        // Elementwise - Shifts
        left_shift,
        right_shift,
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
        atan2,
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
