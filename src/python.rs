//! Python bindings for [`DDSketch`](crate::DDSketch) via PyO3.
//!
//! These bindings are only compiled when the `python` feature is enabled.
//! The exposed Python class is named `DDSketch` (see the `ddsketchy`
//! pymodule).

use crate::ddsketchy::{DDSketch as DDSketchInner, DDSketchError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-facing wrapper around [`crate::DDSketch`].
///
/// Constructed from Python as `DDSketch(alpha=0.01)`.
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
    /// Creates a new sketch with the given relative-error `alpha`.
    ///
    /// Raises `ValueError` if `alpha` is outside `(0, 1)`.
    #[new]
    #[pyo3(signature = (alpha=0.01))]
    fn new(alpha: f64) -> PyResult<Self> {
        let inner = DDSketchInner::new(alpha).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Adds a single value to the sketch.
    fn add(&mut self, value: f64) {
        self.inner.add(value);
    }

    /// Adds every value in the provided sequence to the sketch.
    fn add_batch(&mut self, values: Vec<f64>) {
        self.inner.add_batch(values);
    }

    /// Returns the estimated value at quantile `q`, where `q` is in `[0, 1]`.
    ///
    /// Raises `ValueError` if `q` is outside `[0, 1]`.
    fn quantile(&self, q: f64) -> PyResult<f64> {
        self.inner.quantile(q).map_err(PyErr::from)
    }

    /// Merges another sketch into this one in place.
    ///
    /// Raises `ValueError` if the two sketches use different `alpha` values.
    fn merge(&mut self, other: &DDSketch) -> PyResult<()> {
        self.inner.merge(&other.inner).map_err(PyErr::from)
    }

    /// Number of values added to the sketch.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Sum of all values added to the sketch.
    #[getter]
    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    /// Arithmetic mean of all values, or `0.0` when empty.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Reconstructed minimum value, or `+inf` when empty.
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }

    /// Reconstructed maximum value, or `-inf` when empty.
    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }

    /// The `alpha` relative-error parameter the sketch was built with.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// Returns `True` if no values have been added.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clears all accumulated data, preserving the `alpha` configuration.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns `(P50, P90, P95, P99)`, or `None` if the sketch is empty.
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

/// The `ddsketchy` Python module entry point.
#[pymodule]
fn ddsketchy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DDSketch>()?;
    Ok(())
}
