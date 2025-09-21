#![forbid(unsafe_code)]

mod dd_sketchy;
#[cfg(test)]
mod dd_sketchy_test;

#[cfg(all(test, feature = "serde"))]
mod serde_tests;

pub use dd_sketchy::{DDSketch, DDSketchError, DDSketchBuilder};
