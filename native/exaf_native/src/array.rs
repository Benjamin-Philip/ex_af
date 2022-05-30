use crate::datatypes::*;

use arrayfire::Dim4;
use half::f16;
use num_complex::{Complex32, Complex64};
use rustler::types::{Binary, OwnedBinary};
use rustler::Env;
use std::convert::TryInto;

// Public API

// Creation

#[rustler::nif]
pub fn iota(shape: Vec<u64>, tdims: Vec<u64>, dtype: String) -> ExAf {
    let shape = dim_from_shape(shape);
    let dtype = dtype_from_string(dtype);
    let tdims = dim_from_shape(tdims);

    let exaf_array = match dtype {
        ExAfDType::U8 => arrayfire::iota::<u8>(shape, tdims).to_exaf_array(),
        ExAfDType::U16 => arrayfire::iota::<u16>(shape, tdims).to_exaf_array(),
        ExAfDType::U32 => arrayfire::iota::<u32>(shape, tdims).to_exaf_array(),
        ExAfDType::U64 => arrayfire::iota::<u64>(shape, tdims).to_exaf_array(),
        ExAfDType::S16 => arrayfire::iota::<i16>(shape, tdims).to_exaf_array(),
        ExAfDType::S32 => arrayfire::iota::<i32>(shape, tdims).to_exaf_array(),
        ExAfDType::S64 => arrayfire::iota::<i64>(shape, tdims).to_exaf_array(),
        ExAfDType::F16 => arrayfire::iota::<f16>(shape, tdims).to_exaf_array(),
        ExAfDType::F32 => arrayfire::iota::<f32>(shape, tdims).to_exaf_array(),
        ExAfDType::F64 => arrayfire::iota::<f64>(shape, tdims).to_exaf_array(),
        ExAfDType::C64 => arrayfire::iota::<Complex32>(shape, tdims).to_exaf_array(),
        ExAfDType::C128 => arrayfire::iota::<Complex64>(shape, tdims).to_exaf_array(),
    };

    ExAf::from_exaf_array(exaf_array)
}

// Conversion

#[rustler::nif]
pub fn from_binary(binary: Binary, shape: Vec<u64>, dtype: String) -> ExAf {
    let dim = dim_from_shape(shape);
    let slice = binary.as_slice();
    let dtype = dtype_from_string(dtype);

    ExAf::from_slice(slice, dim, dtype)
}

#[rustler::nif]
pub fn to_binary(env: Env, array: ExAf, limit: usize) -> Binary {
    let exaf_array = array.resource.value();

    let mut vec = exaf_array.to_vec();
    vec.truncate(exaf_array.dtype().bytes() * limit);

    let slice = vec.as_slice();

    let mut erl_bin = OwnedBinary::new(vec.len()).unwrap();
    erl_bin.as_mut_slice().copy_from_slice(slice);

    erl_bin.release(env)
}

// Elementwise

macro_rules! unary_op {
    ($op_name:ident, $af_op:ident) => {
        #[rustler::nif]
        pub fn $op_name(array: ExAf) -> ExAf {
            let ex_array = array.resource.value();
            apply_function_array!(ex_array, $af_op,)
        }
    };
}

// Elementwise - Exponentation

unary_op!(exp, exp);
unary_op!(expm1, expm1);
unary_op!(log, log);
unary_op!(log1p, log1p);
unary_op!(sigmoid, sigmoid);

// Elementwise - Trignometry

unary_op!(sin, sin);
unary_op!(cos, cos);
unary_op!(tan, tan);
unary_op!(sinh, sinh);
unary_op!(cosh, cosh);
unary_op!(tanh, tanh);
unary_op!(asin, asin);
unary_op!(acos, acos);
unary_op!(atan, atan);
unary_op!(asinh, asinh);
unary_op!(acosh, acosh);
unary_op!(atanh, atanh);

// Elementwise - Error Functions

unary_op!(erf, erf);
unary_op!(erfc, erfc);

// Elementwise - Error Functions

unary_op!(sqrt, sqrt);
unary_op!(rsqrt, rsqrt);
unary_op!(cbrt, cbrt);

// Elementwise - Number Theoryesque Functions

unary_op!(abs, abs);
unary_op!(floor, floor);
unary_op!(round, round);
unary_op!(sign, sign);
unary_op!(ceil, ceil);
unary_op!(real, real);
unary_op!(imag, imag);

// Shape

#[rustler::nif]
pub fn reshape(array: ExAf, shape: Vec<u64>) -> ExAf {
    let dim = dim_from_shape(shape);
    let exaf_array = array.resource.value();

    apply_function_array!(exaf_array, moddims, dim)
}

// Type

#[rustler::nif]
pub fn as_type(array: ExAf, dtype: String) -> ExAf {
    let dtype = dtype_from_string(dtype);
    let exaf_array = array.resource.value();

    // Complex arrays don't support casting.
    // So take their real part and then cast.

    // TODO: Use the local real function instead

    let exaf_array = match exaf_array {
        ExAfArray::C64(ref a) => ExAfArray::F32(arrayfire::real(a)),
        ExAfArray::C128(ref a) => ExAfArray::F64(arrayfire::real(a)),
        _ => exaf_array,
    };

    let new_exaf_array = match dtype {
        ExAfDType::U8 => ExAfArray::U8(apply_generic_method_array!(exaf_array, cast, u8,)),
        ExAfDType::U16 => ExAfArray::U16(apply_generic_method_array!(exaf_array, cast, u16,)),
        ExAfDType::U32 => ExAfArray::U32(apply_generic_method_array!(exaf_array, cast, u32,)),
        ExAfDType::U64 => ExAfArray::U64(apply_generic_method_array!(exaf_array, cast, u64,)),
        ExAfDType::S16 => ExAfArray::S16(apply_generic_method_array!(exaf_array, cast, i16,)),
        ExAfDType::S32 => ExAfArray::S32(apply_generic_method_array!(exaf_array, cast, i32,)),
        ExAfDType::S64 => ExAfArray::S64(apply_generic_method_array!(exaf_array, cast, i64,)),
        ExAfDType::F16 => ExAfArray::F16(apply_generic_method_array!(exaf_array, cast, f16,)),
        ExAfDType::F32 => ExAfArray::F32(apply_generic_method_array!(exaf_array, cast, f32,)),
        ExAfDType::F64 => ExAfArray::F64(apply_generic_method_array!(exaf_array, cast, f64,)),
        ExAfDType::C64 => ExAfArray::C64(apply_generic_method_array!(exaf_array, cast, Complex32,)),
        ExAfDType::C128 => {
            ExAfArray::C128(apply_generic_method_array!(exaf_array, cast, Complex64,))
        }
    };

    ExAf::from_exaf_array(new_exaf_array)
}

// Helpers

fn dim_from_shape(shape: Vec<u64>) -> Dim4 {
    let array: [u64; 4] = match shape.as_slice().try_into() {
        Ok(ba) => ba,
        Err(_) => panic!(
            "Expected a shape of length {} but it was {}",
            32,
            shape.len()
        ),
    };
    Dim4::new(&array)
}

fn dtype_from_string(dtype: String) -> ExAfDType {
    match dtype.as_str() {
        "u8" => ExAfDType::U8,
        "u16" => ExAfDType::U16,
        "u32" => ExAfDType::U32,
        "u64" => ExAfDType::U64,
        "s16" => ExAfDType::S16,
        "s32" => ExAfDType::S32,
        "s64" => ExAfDType::S64,
        "f16" => ExAfDType::F16,
        "f32" => ExAfDType::F32,
        "f64" => ExAfDType::F64,
        "c64" => ExAfDType::C64,
        "c128" => ExAfDType::C128,
        _ => unimplemented!(),
    }
}
