mod random;
mod storage;
mod matrix;
mod kernel;
mod moments;
mod fitter;
mod observation;
use pyo3::prelude::*;
#[pymodule]
fn _native(m:&Bound<'_,PyModule>)->PyResult<()>{
    m.add_class::<storage::ArrayView>()?;
    m.add_class::<kernel::NativeKernel>()?;
    m.add_class::<fitter::NativeFitter>()?;
    m.add_class::<observation::NativeObservation>()?;
    m.add_class::<observation::NativeModel>()?;
    m.add_function(wrap_pyfunction!(moments::moments,m)?)?;
    m.add_function(wrap_pyfunction!(moments::probability,m)?)?;
    m.add_function(wrap_pyfunction!(moments::loglike,m)?)?;
    m.add_function(wrap_pyfunction!(moments::normpdf,m)?)?;
    m.add_function(wrap_pyfunction!(moments::normcdf,m)?)?;
    m.add_function(wrap_pyfunction!(moments::logphi,m)?)?;
    m.add_function(wrap_pyfunction!(moments::log_factorial,m)?)?;
    m.add_function(wrap_pyfunction!(moments::logsumexp,m)?)?;
    m.add_function(wrap_pyfunction!(moments::logsumexp2,m)?)?;
    m.add_function(wrap_pyfunction!(moments::iv,m)?)?;
    m.add_function(wrap_pyfunction!(moments::distances,m)?)?;
    m.add_function(wrap_pyfunction!(observation::f_params,m)?)?;
    Ok(())
}
