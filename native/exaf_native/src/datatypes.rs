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
    pub fn from_slice(slice: &[u8], dim: Dim4, dtype: ExAfDType) -> Self {
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

pub trait ArrayToExAfArray {
    fn to_exaf_array(&self) -> ExAfArray;
}

impl ArrayToExAfArray for Array<bool> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::U8(self.cast::<u8>().copy())
    }
}

impl ArrayToExAfArray for Array<u8> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::U8(self.copy())
    }
}

impl ArrayToExAfArray for Array<u16> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::U16(self.copy())
    }
}

impl ArrayToExAfArray for Array<u32> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::U32(self.copy())
    }
}

impl ArrayToExAfArray for Array<u64> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::U64(self.copy())
    }
}

impl ArrayToExAfArray for Array<i16> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::S16(self.copy())
    }
}

impl ArrayToExAfArray for Array<i32> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::S32(self.copy())
    }
}

impl ArrayToExAfArray for Array<i64> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::S64(self.copy())
    }
}

impl ArrayToExAfArray for Array<f16> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::F16(self.copy())
    }
}

impl ArrayToExAfArray for Array<f32> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::F32(self.copy())
    }
}

impl ArrayToExAfArray for Array<f64> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::F64(self.copy())
    }
}

impl ArrayToExAfArray for Array<Complex32> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::C64(self.copy())
    }
}

impl ArrayToExAfArray for Array<Complex64> {
    fn to_exaf_array(&self) -> ExAfArray {
        ExAfArray::C128(self.copy())
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
macro_rules! apply_generic_method_array {
    ($self:ident, $method:ident, $type:ident, $($args:expr),*) => {
        match $self {
            ExAfArray::U8(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::U16(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::U32(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::U64(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::S16(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::S32(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::S64(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::F16(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::F32(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::F64(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::C64(ref a) => a.$method::<$type>($($args), *),
            ExAfArray::C128(ref a) => a.$method::<$type>($($args), *),
        }
    };
}

pub(crate) use apply_generic_method_array;

#[macro_export]
macro_rules! apply_function_array {
    ($self:ident, $function:ident, $($args:expr),*) => {
        ExAf::from_exaf_array( match $self {
            ExAfArray::U8(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::U16(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::U32(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::U64(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::S16(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::S32(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::S64(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::F16(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::F32(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::F64(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::C64(ref a) => arrayfire::$function(a, $($args), *).to_exaf_array(),
            ExAfArray::C128(ref a) =>arrayfire::$function(a, $($args), *).to_exaf_array(),
        })
    };
}

pub(crate) use apply_function_array;

#[macro_export]
macro_rules! apply_generic_function_array {
    ($function:ident, $dtype:ident, $($args:expr),*) => {
        ExAf::from_exaf_array(match $dtype {
            ExAfDType::U8 => arrayfire::$function::<u8>($($args), *).to_exaf_array(),
            ExAfDType::U16 => arrayfire::$function::<u16>($($args), *).to_exaf_array(),
            ExAfDType::U32 => arrayfire::$function::<u32>($($args), *).to_exaf_array(),
            ExAfDType::U64 => arrayfire::$function::<u64>($($args), *).to_exaf_array(),
            ExAfDType::S16 => arrayfire::$function::<i16>($($args), *).to_exaf_array(),
            ExAfDType::S32 => arrayfire::$function::<i32>($($args), *).to_exaf_array(),
            ExAfDType::S64 => arrayfire::$function::<i64>($($args), *).to_exaf_array(),
            ExAfDType::F16 => arrayfire::$function::<f16>($($args), *).to_exaf_array(),
            ExAfDType::F32 => arrayfire::$function::<f32>($($args), *).to_exaf_array(),
            ExAfDType::F64 => arrayfire::$function::<f64>($($args), *).to_exaf_array(),
            ExAfDType::C64 => arrayfire::$function::<Complex32>($($args), *).to_exaf_array(),
            ExAfDType::C128 => arrayfire::$function::<Complex64>($($args), *).to_exaf_array(),
        })
    };
}

pub(crate) use apply_generic_function_array;

pub struct ExAfRef(pub RwLock<ExAfArray>);

#[derive(NifStruct)]
#[module = "ExAF.Backend"]
pub struct ExAf {
    pub resource: ResourceArc<ExAfRef>,
}

impl ExAfRef {
    pub fn from_exaf_array(array: ExAfArray) -> Self {
        Self(RwLock::new(array))
    }

    pub fn from_slice(slice: &[u8], dim: Dim4, dtype: ExAfDType) -> Self {
        Self(RwLock::new(ExAfArray::from_slice(slice, dim, dtype)))
    }

    pub fn value(&self) -> ExAfArray {
        match self.0.try_read() {
            Ok(reference) => reference.clone(),
            Err(_) => unreachable!(),
        }
    }
}

impl ExAf {
    pub fn from_exaf_array(array: ExAfArray) -> Self {
        Self {
            resource: ResourceArc::new(ExAfRef::from_exaf_array(array)),
        }
    }
    pub fn from_slice(slice: &[u8], dim: Dim4, dtype: ExAfDType) -> Self {
        Self {
            resource: ResourceArc::new(ExAfRef::from_slice(slice, dim, dtype)),
        }
    }
}
