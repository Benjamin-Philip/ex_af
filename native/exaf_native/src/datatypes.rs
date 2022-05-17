use arrayfire::{Array, Dim4};
use half::f16;
use num_complex::{Complex, Complex32, Complex64};
use rustler::resource::ResourceArc;
use rustler::NifStruct;
use std::sync::RwLock;

#[derive(PartialEq)]
pub enum ExAfDType {
    U8,
    U16,
    U32,
    U64,
    S16,
    S32,
    S64,
    F16,
    F32,
    F64,
    C64,
    C128,
}

impl ExAfDType {
    pub fn bytes(&self) -> usize {
        match self {
            ExAfDType::U8 => 1,
            ExAfDType::U16 | ExAfDType::S16 | ExAfDType::F16 => 2,
            ExAfDType::U32 | ExAfDType::S32 | ExAfDType::F32 => 4,
            ExAfDType::U64 | ExAfDType::S64 | ExAfDType::F64 | ExAfDType::C64 => 8,
            ExAfDType::C128 => 16,
        }
    }
}

#[derive(Clone)]
pub enum ExAfArray {
    U8(Array<u8>),
    U16(Array<u16>),
    U32(Array<u32>),
    U64(Array<u64>),
    S16(Array<i16>),
    S32(Array<i32>),
    S64(Array<i64>),
    F16(Array<f16>),
    F32(Array<f32>),
    F64(Array<f64>),
    C64(Array<Complex32>),
    C128(Array<Complex64>),
}

impl ExAfArray {
    pub fn new(slice: &[u8], dim: Dim4, dtype: ExAfDType) -> Self {
        match dtype {
            ExAfDType::U8 => ExAfArray::U8(Array::new(slice, dim)),
            ExAfDType::U16 => {
                ExAfArray::U16(Array::new(unsafe { &(*slice.align_to::<u16>().1) }, dim))
            }
            ExAfDType::U32 => {
                ExAfArray::U32(Array::new(unsafe { &(*slice.align_to::<u32>().1) }, dim))
            }
            ExAfDType::U64 => {
                ExAfArray::U64(Array::new(unsafe { &(*slice.align_to::<u64>().1) }, dim))
            }
            ExAfDType::S16 => {
                ExAfArray::S16(Array::new(unsafe { &(*slice.align_to::<i16>().1) }, dim))
            }
            ExAfDType::S32 => {
                ExAfArray::S32(Array::new(unsafe { &(*slice.align_to::<i32>().1) }, dim))
            }
            ExAfDType::S64 => {
                ExAfArray::S64(Array::new(unsafe { &(*slice.align_to::<i64>().1) }, dim))
            }
            ExAfDType::F16 => {
                ExAfArray::F16(Array::new(unsafe { &(*slice.align_to::<f16>().1) }, dim))
            }
            ExAfDType::F32 => {
                ExAfArray::F32(Array::new(unsafe { &(*slice.align_to::<f32>().1) }, dim))
            }
            ExAfDType::F64 => {
                ExAfArray::F64(Array::new(unsafe { &(*slice.align_to::<f64>().1) }, dim))
            }

            ExAfDType::C64 => {
                let mut complex_vec = Vec::new();
                for chunk in unsafe { slice.align_to::<f32>().1.chunks_exact(2) } {
                    let mut chunk_iter = chunk.iter();

                    let re = chunk_iter.next().unwrap();
                    let im = chunk_iter.next().unwrap();

                    complex_vec.push(Complex::new(*re, *im))
                }

                ExAfArray::C64(Array::new(complex_vec.as_slice(), dim))
            }

            ExAfDType::C128 => {
                let mut complex_vec = Vec::new();
                for chunk in unsafe { slice.align_to::<f64>().1.chunks_exact(2) } {
                    let mut chunk_iter = chunk.iter();

                    let re = chunk_iter.next().unwrap();
                    let im = chunk_iter.next().unwrap();

                    complex_vec.push(Complex::new(*re, *im))
                }

                ExAfArray::C128(Array::new(complex_vec.as_slice(), dim))
            }
        }
    }

    pub fn to_vec(&self) -> Vec<u8> {
        let nelements = apply_method_array!(self, elements,);

        match self.dtype() {
            ExAfDType::U8 => {
                let mut vector = vec![u8::default(); nelements];

                apply_method_array!(self, host, &mut vector);
                vector
            }
            ExAfDType::U16 => {
                let mut vector = vec![u16::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::U32 => {
                let mut vector = vec![u32::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::U64 => {
                let mut vector = vec![u64::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::S16 => {
                let mut vector = vec![i16::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::S32 => {
                let mut vector = vec![i32::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::S64 => {
                let mut vector = vec![i64::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::F16 => {
                let mut vector = vec![f16::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }

            ExAfDType::F32 => {
                let mut vector = vec![f32::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::F64 => {
                let mut vector = vec![f64::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::C64 => {
                let mut vector = vec![Complex32::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
            ExAfDType::C128 => {
                let mut vector = vec![Complex64::default(); nelements];
                apply_method_array!(self, host, &mut vector);

                unsafe { vector.align_to::<u8>().1.to_vec() }
            }
        }
    }

    pub fn dtype(&self) -> ExAfDType {
        match self {
            ExAfArray::U8(_a) => ExAfDType::U8,
            ExAfArray::U16(_a) => ExAfDType::U16,
            ExAfArray::U32(_a) => ExAfDType::U32,
            ExAfArray::U64(_a) => ExAfDType::U64,
            ExAfArray::S16(_a) => ExAfDType::S16,
            ExAfArray::S32(_a) => ExAfDType::S32,
            ExAfArray::S64(_a) => ExAfDType::S64,
            ExAfArray::F16(_a) => ExAfDType::F16,
            ExAfArray::F32(_a) => ExAfDType::F32,
            ExAfArray::F64(_a) => ExAfDType::F64,
            ExAfArray::C64(_a) => ExAfDType::C64,
            ExAfArray::C128(_a) => ExAfDType::C128,
        }
    }
}

#[macro_export]
macro_rules! apply_method_array {
    ($self:ident, $method:ident, $($args:expr),*) => {
        match $self {
            ExAfArray::U8(ref a) => a.$method($($args), *),
            ExAfArray::U16(ref a) => a.$method($($args), *),
            ExAfArray::U32(ref a) => a.$method($($args), *),
            ExAfArray::U64(ref a) => a.$method($($args), *),
            ExAfArray::S16(ref a) => a.$method($($args), *),
            ExAfArray::S32(ref a) => a.$method($($args), *),
            ExAfArray::S64(ref a) => a.$method($($args), *),
            ExAfArray::F16(ref a) => a.$method($($args), *),
            ExAfArray::F32(ref a) => a.$method($($args), *),
            ExAfArray::F64(ref a) => a.$method($($args), *),
            ExAfArray::C64(ref a) => a.$method($($args), *),
            ExAfArray::C128(ref a) => a.$method($($args), *),
        }
    };
}

pub(crate) use apply_method_array;

#[macro_export]
macro_rules! apply_function_array {
    ($self:ident, $method:ident, $($args:expr),*) => {
        match $self {
            ExAfArray::U8(ref a) => $method(a, $($args), *),
            ExAfArray::U16(ref a) => $method(a, $($args), *),
            ExAfArray::U32(ref a) => $method(a, $($args), *),
            ExAfArray::U64(ref a) => $method(a, $($args), *),
            ExAfArray::S16(ref a) => $method(a, $($args), *),
            ExAfArray::S32(ref a) => $method(a, $($args), *),
            ExAfArray::S64(ref a) => $method(a, $($args), *),
            ExAfArray::F16(ref a) => $method(a, $($args), *),
            ExAfArray::F32(ref a) => $method(a, $($args), *),
            ExAfArray::F64(ref a) => $method(a, $($args), *),
            ExAfArray::C64(ref a) => $method(a, $($args), *),
            ExAfArray::C128(ref a) => $method(a, $($args), *),
        }
    };
}

pub(crate) use apply_function_array;

pub struct ExAfRef(pub RwLock<ExAfArray>);

#[derive(NifStruct)]
#[module = "ExAF.Backend"]
pub struct ExAf {
    pub resource: ResourceArc<ExAfRef>,
}

impl ExAfRef {
    pub fn new(slice: &[u8], dim: Dim4, dtype: ExAfDType) -> Self {
        Self(RwLock::new(ExAfArray::new(slice, dim, dtype)))
    }

    pub fn value(&self) -> ExAfArray {
        match self.0.try_read() {
            Ok(reference) => reference.clone(),
            Err(_) => unreachable!(),
        }
    }
}

impl ExAf {
    pub fn new(slice: &[u8], dim: Dim4, dtype: ExAfDType) -> Self {
        Self {
            resource: ResourceArc::new(ExAfRef::new(slice, dim, dtype)),
        }
    }
}
