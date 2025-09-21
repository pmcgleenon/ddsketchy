#![forbid(unsafe_code)]

mod dd_sketchy;
#[cfg(test)]
mod dd_sketchy_test;

#[cfg(feature = "serde")]
#[cfg(test)]
mod serde_tests;

pub use dd_sketchy::{DDSketch, DDSketchError, DDSketchBuilder};
