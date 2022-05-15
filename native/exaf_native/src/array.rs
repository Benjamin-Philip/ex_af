use crate::datatypes::*;

use arrayfire::Dim4;
use rustler::types::{Binary, OwnedBinary};
use rustler::Env;
use std::convert::TryInto;

// Public API

#[rustler::nif]
pub fn from_binary(binary: Binary, shape: Vec<u64>, dtype: String) -> ExAf {
    let dim = dim_from_shape(shape);
    let slice = binary.as_slice();
    let dtype = dtype_from_string(dtype);

    ExAf::new(slice, dim, dtype)
}

#[rustler::nif]
pub fn to_binary(env: Env, array: ExAf) -> Binary {
    let ex_array = array.resource.value();

    let vec = ex_array.to_vec();
    let slice = vec.as_slice();

    let mut erl_bin = OwnedBinary::new(vec.len()).unwrap();
    erl_bin.as_mut_slice().copy_from_slice(slice);

    erl_bin.release(env)
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
        "f32" => ExAfDType::F32,
        "f64" => ExAfDType::F64,
        _ => unimplemented!(),
    }
}