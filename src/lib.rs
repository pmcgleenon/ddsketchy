#![forbid(unsafe_code)]

mod dd_sketchy;

#[cfg(test)]
mod dd_sketchy_test;

#[cfg(test)]
mod datadog_reference_tests;




#[cfg(feature = "serde")]
#[cfg(test)]
mod serde_tests;

pub use dd_sketchy::{DDSketch, DDSketchBuilder, DDSketchError};
