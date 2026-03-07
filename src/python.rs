use crate::ddsketchy::{DDSketch as DDSketchInner, DDSketchError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct DDSketch {
    inner: DDSketchInner,
}

impl From<DDSketchError> for PyErr {
    fn from(err: DDSketchError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

#[pymethods]
impl DDSketch {
    #[new]
    #[pyo3(signature = (alpha=0.01))]
    fn new(alpha: f64) -> PyResult<Self> {
        let inner = DDSketchInner::new(alpha).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    fn add(&mut self, value: f64) {
        self.inner.add(value);
    }

    fn add_batch(&mut self, values: &Bound<'_, PyList>) -> PyResult<()> {
        for value in values {
            self.inner.add(value.extract()?);
        }
        Ok(())
    }

    fn quantile(&self, q: f64) -> PyResult<f64> {
        self.inner.quantile(q).map_err(PyErr::from)
    }

    fn merge(&mut self, other: &DDSketch) -> PyResult<()> {
        self.inner.merge(&other.inner).map_err(PyErr::from)
    }

    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    #[getter]
    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }

    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn percentiles(&self) -> Option<(f64, f64, f64, f64)> {
        self.inner.percentiles()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pymodule(name = "ddsketchy")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DDSketch>()?;
    Ok(())
}
