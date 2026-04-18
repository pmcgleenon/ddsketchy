#![forbid(unsafe_code)]
//! # ddsketchy
//!
//! A Rust implementation of the [DDSketch] quantile sketch algorithm.
//!
//! DDSketch is a fast, fully-mergeable quantile sketch with **relative-error
//! guarantees**: every quantile estimate is within a user-chosen multiplicative
//! factor `alpha` of the true value. This makes it well-suited to latency
//! distributions and other data that spans many orders of magnitude, where
//! absolute-error sketches behave poorly. Sketches are cheap to merge, making
//! them a natural fit for distributed aggregation.
//!
//! [DDSketch]: https://arxiv.org/pdf/1908.10693.pdf
//!
//! # Quick start
//!
//! ```
//! use ddsketchy::DDSketch;
//!
//! let mut sketch = DDSketch::new(0.01).expect("valid alpha");
//! for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
//!     sketch.add(v);
//! }
//!
//! // Quantiles are accurate to within `alpha` (1%) relative error.
//! let median = sketch.quantile(0.5).unwrap();
//! assert!((median - 3.0).abs() < 3.0 * 0.01 + 0.05);
//! assert_eq!(sketch.count(), 5);
//! ```
//!
//! # Features
//!
//! - `serde` — enables `Serialize` / `Deserialize` for [`DDSketch`].
//! - `python` — enables PyO3 bindings (used by the Python wheel build).
//!
//! # More examples
//!
//! See the [`examples/`] directory in the repository for runnable examples,
//! including serialization round-trips and the code snippets shown in the
//! README.
//!
//! [`examples/`]: https://github.com/pmcgleenon/ddsketchy/tree/main/examples

mod ddsketchy;
mod mapping;
mod store;

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod ddsketchy_test;

#[cfg(test)]
mod datadog_reference_tests;

#[cfg(feature = "serde")]
#[cfg(test)]
mod serde_tests;

pub use ddsketchy::{DDSketch, DDSketchBuilder, DDSketchError};
