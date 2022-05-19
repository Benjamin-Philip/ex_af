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
    [from_binary, to_binary, reshape],
    load = load
);
