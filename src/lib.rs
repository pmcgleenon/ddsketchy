#![forbid(unsafe_code)]

mod dd_sketchy;
#[cfg(test)]
mod dd_sketchy_test;

#[cfg(feature = "serde")]
#[cfg(test)]
mod serde_tests;

#[cfg(test)]
mod datadog_reference_tests;

#[cfg(test)]
mod debug_single_value;

#[cfg(test)]
mod debug_boundary;

#[cfg(test)]
mod debug_zero_negative;

pub use dd_sketchy::{DDSketch, DDSketchBuilder, DDSketchError};
