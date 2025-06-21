#![forbid(unsafe_code)]

mod dd_sketchy;
#[cfg(test)]
mod dd_sketchy_test;

pub use dd_sketchy::{DDSketch, DDSketchError, DDSketchBuilder};
