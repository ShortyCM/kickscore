use pyo3::{prelude::*, types::PyDict};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{cell::UnsafeCell, ops::{Deref, DerefMut}, rc::Rc};

pub struct Array<T> { data: Rc<UnsafeCell<Vec<T>>> }
impl<T> Array<T> {
    pub fn from_vec(values: Vec<T>) -> Self { Self { data: Rc::new(UnsafeCell::new(values)) } }
    pub fn new() -> Self { Self::from_vec(Vec::new()) }
}
impl<T: Clone> Array<T> {
    pub fn reserve(&mut self, additional: usize) {
        if Rc::strong_count(&self.data) > 1 {
            self.data = Rc::new(UnsafeCell::new(self.deref().clone()));
        }
        unsafe { &mut *self.data.get() }.reserve(additional);
    }

    pub fn push(&mut self, value: T) {
        if Rc::strong_count(&self.data) > 1 {
            self.data = Rc::new(UnsafeCell::new(self.deref().clone()));
        }
        unsafe { &mut *self.data.get() }.push(value);
    }
    pub fn clear(&mut self) {
        if Rc::strong_count(&self.data) > 1 { self.data = Rc::new(UnsafeCell::new(Vec::new())); }
        else { unsafe { &mut *self.data.get() }.clear(); }
    }
}
impl<T: Clone> Clone for Array<T> {
    fn clone(&self) -> Self { Self::from_vec(self.deref().clone()) }
}
impl<T> Deref for Array<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> { unsafe { &*self.data.get() } }
}
impl<T> DerefMut for Array<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> { unsafe { &mut *self.data.get() } }
}
impl<T: Serialize> Serialize for Array<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> { self.deref().serialize(serializer) }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Array<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> { Ok(Self::from_vec(Vec::<T>::deserialize(deserializer)?)) }
}
#[pyclass(unsendable)]
pub struct ArrayView { storage: ViewStorage }
enum ViewStorage { Float(Rc<UnsafeCell<Vec<f64>>>), Integer(Rc<UnsafeCell<Vec<i64>>>) }
impl Array<f64> {
    pub fn prepare_output(&mut self, n: usize) {
        if Rc::strong_count(&self.data) > 1 { self.data = Rc::new(UnsafeCell::new(vec![0.0; n])); }
        else { unsafe { &mut *self.data.get() }.resize(n, 0.0); }
    }
    pub fn view(&self) -> ArrayView { ArrayView { storage: ViewStorage::Float(self.data.clone()) } } }
impl Array<i64> { pub fn view(&self) -> ArrayView { ArrayView { storage: ViewStorage::Integer(self.data.clone()) } } }
#[pymethods]
impl ArrayView {
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (pointer, len, kind) = match &self.storage {
            ViewStorage::Float(data) => { let values = unsafe { &*data.get() }; (values.as_ptr() as usize, values.len(), "f8") },
            ViewStorage::Integer(data) => { let values = unsafe { &*data.get() }; (values.as_ptr() as usize, values.len(), "i8") },
        };
        let result = PyDict::new(py);
        result.set_item("version", 3)?;
        result.set_item("shape", (len,))?;
        result.set_item("typestr", format!("{}{}", if cfg!(target_endian = "little") { "<" } else { ">" }, kind))?;
        result.set_item("data", (pointer, false))?;
        Ok(result)
    }
}
