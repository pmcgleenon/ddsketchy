#![forbid(unsafe_code)]

mod ddsketchy;
mod mapping;
mod store;

#[cfg(test)]
mod ddsketchy_test;

#[cfg(test)]
mod datadog_reference_tests;

#[cfg(feature = "serde")]
#[cfg(test)]
mod serde_tests;

pub use ddsketchy::{DDSketch, DDSketchBuilder, DDSketchError};
