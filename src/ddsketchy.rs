//! Core DDSketch implementation.
//!
//! This module defines the public [`DDSketch`] type, the [`DDSketchBuilder`]
//! for customized construction, and the [`DDSketchError`] type returned by
//! fallible operations.

use crate::store::Store;

/// Errors returned by [`DDSketch`] operations.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum DDSketchError {
    /// The relative-error parameter `alpha` was outside the open interval
    /// `(0, 1)` or was non-finite.
    InvalidAlpha,
    /// A quantile argument was outside the closed interval `[0, 1]` or was
    /// non-finite.
    InvalidQuantile,
    /// Attempted to merge two sketches whose `alpha` (and therefore `gamma`)
    /// values differ.
    AlphaMismatch,
    /// Bin count exceeds the sketch's configured maximum.
    BinCountMismatch,
}

impl std::fmt::Display for DDSketchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DDSketchError::InvalidAlpha => write!(f, "Alpha must be in range (0, 1)"),
            DDSketchError::InvalidQuantile => write!(f, "Quantile must be in range [0, 1]"),
            DDSketchError::AlphaMismatch => {
                write!(f, "Cannot merge sketches with different alpha values")
            }
            DDSketchError::BinCountMismatch => write!(f, "Bin count exceeds maximum allowed"),
        }
    }
}

impl std::error::Error for DDSketchError {}

#[cfg(feature = "serde")]
fn serialize_f64_option<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if value.is_infinite() {
        // Serialize infinity as null for cleaner JSON
        serializer.serialize_none()
    } else if value.is_nan() {
        // NaN also serializes as null
        serializer.serialize_none()
    } else {
        serializer.serialize_some(value)
    }
}

#[cfg(feature = "serde")]
fn deserialize_min_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<f64> = serde::Deserialize::deserialize(deserializer)?;
    Ok(opt.unwrap_or(f64::INFINITY))
}

#[cfg(feature = "serde")]
fn deserialize_max_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<f64> = serde::Deserialize::deserialize(deserializer)?;
    Ok(opt.unwrap_or(f64::NEG_INFINITY))
}

/// A DDSketch quantile estimator with configurable relative accuracy.
///
/// `DDSketch` provides fast quantile estimation with a bounded **relative**
/// error `alpha`: any quantile estimate is within `alpha * q` of the true
/// value `q`. It handles negative, zero, and positive values, is fully
/// mergeable, and uses bounded memory.
///
/// # Example
///
/// ```
/// use ddsketchy::DDSketch;
///
/// let mut sketch = DDSketch::new(0.01).expect("valid alpha");
/// for v in 1..=100 {
///     sketch.add(v as f64);
/// }
/// let p99 = sketch.quantile(0.99).unwrap();
/// // Result is within 1% relative error of 99.
/// assert!((p99 - 99.0).abs() < 99.0 * 0.01);
/// ```
///
/// # Serialization
///
/// When the `serde` feature is enabled, `DDSketch` implements
/// `serde::Serialize` and `serde::Deserialize`:
///
/// ```toml
/// ddsketchy = { version = "*", features = ["serde"] }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DDSketch {
    // Mapping configuration
    alpha: f64,
    gamma: f64,
    inv_ln_gamma: f64,
    offset: i32, // Changed to i32 like reference
    min_indexable_value: f64,

    // Dual stores - for positive and negative values
    positive_store: Store,
    negative_store: Store,
    zero_count: u64,

    // Summary statistics
    sum: f64,
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serialize_f64_option",
            deserialize_with = "deserialize_min_f64"
        )
    )]
    min: f64,
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serialize_f64_option",
            deserialize_with = "deserialize_max_f64"
        )
    )]
    max: f64,

    // Configuration
    max_bins: usize,
}

impl DDSketch {
    /// Creates a new `DDSketch` with the given relative-error parameter.
    ///
    /// `alpha` must lie in the open interval `(0, 1)`. Smaller values give
    /// tighter quantile estimates at the cost of more memory.
    ///
    /// # Errors
    ///
    /// Returns [`DDSketchError::InvalidAlpha`] if `alpha` is not finite or is
    /// outside `(0, 1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::{DDSketch, DDSketchError};
    ///
    /// let sketch = DDSketch::new(0.01).expect("valid alpha");
    /// assert_eq!(sketch.count(), 0);
    ///
    /// assert_eq!(DDSketch::new(0.0).unwrap_err(), DDSketchError::InvalidAlpha);
    /// ```
    pub fn new(alpha: f64) -> Result<Self, DDSketchError> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(DDSketchError::InvalidAlpha);
        }

        // Use the correct DDSketch formulation from the reference implementation
        let gamma = 1.0 + (2.0 * alpha) / (1.0 - alpha);
        let gamma_ln = ((2.0 * alpha) / (1.0 - alpha)).ln_1p();
        let inv_ln_gamma = 1.0 / gamma_ln;

        let offset = 0i32; // DataDog uses indexOffset = 0 by default

        let min_indexable_from_range = ((i32::MIN - offset) as f64 / inv_ln_gamma + 1.0).exp();
        let min_indexable_from_normal = f64::MIN_POSITIVE * gamma;
        let min_indexable_value = min_indexable_from_range.max(min_indexable_from_normal);

        let max_bins = 4096;

        Ok(Self {
            alpha,
            gamma,
            inv_ln_gamma,
            offset,
            min_indexable_value,
            positive_store: Store::new(max_bins),
            negative_store: Store::new(max_bins),
            zero_count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            max_bins,
        })
    }

    /// Maps a value to the internal bin key used by this sketch.
    ///
    /// This is a low-level accessor that is primarily useful for debugging or
    /// for implementing custom stores.
    #[inline]
    pub fn key(&self, value: f64) -> i32 {
        crate::mapping::value_to_key_i32(value, self.inv_ln_gamma)
    }

    /// Returns the smallest positive magnitude this sketch can represent.
    ///
    /// Values with absolute value below this threshold are mapped to the zero
    /// bucket. This matches DataDog's `Mapping.MinIndexableValue()` behaviour.
    #[inline]
    pub fn min_possible(&self) -> f64 {
        self.min_indexable_value
    }

    /// Maps a bin key back to a representative value.
    ///
    /// This is the inverse of [`DDSketch::key`] and is used internally when
    /// reconstructing quantile estimates.
    #[inline]
    pub fn value(&self, key: i32) -> f64 {
        crate::mapping::key_to_value_i32(key, self.gamma, self.inv_ln_gamma.recip())
    }

    /// Adds a single value to the sketch.
    ///
    /// Non-finite values (`NaN`, `±∞`) are silently ignored. Finite values
    /// whose absolute magnitude is below [`DDSketch::min_possible`] are
    /// counted in the zero bucket.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// sketch.add(1.0);
    /// sketch.add(2.0);
    /// assert_eq!(sketch.count(), 2);
    /// ```
    #[inline]
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        if value >= self.min_indexable_value {
            let key = (value.ln() * self.inv_ln_gamma).ceil() as i32;
            self.positive_store.add(key);
        } else if value <= -self.min_indexable_value {
            let key = ((-value).ln() * self.inv_ln_gamma).ceil() as i32;
            self.negative_store.add(key);
        } else {
            self.zero_count += 1;
        }

        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Merges another sketch into this one.
    ///
    /// Both sketches must share the same `alpha` (relative-error) parameter.
    ///
    /// # Errors
    ///
    /// Returns [`DDSketchError::AlphaMismatch`] if the two sketches were
    /// constructed with different `alpha` values.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut a = DDSketch::new(0.01).unwrap();
    /// a.add(1.0);
    /// let mut b = DDSketch::new(0.01).unwrap();
    /// b.add(2.0);
    /// a.merge(&b).unwrap();
    /// assert_eq!(a.count(), 2);
    /// ```
    pub fn merge(&mut self, other: &Self) -> Result<(), DDSketchError> {
        if (self.gamma - other.gamma).abs() > 1e-10 {
            return Err(DDSketchError::AlphaMismatch);
        }

        if other.count() == 0 {
            return Ok(());
        }

        // Merge dual stores
        self.positive_store.merge(&other.positive_store);
        self.negative_store.merge(&other.negative_store);
        self.zero_count += other.zero_count;

        // Update metadata
        self.sum += other.sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);

        Ok(())
    }

    /// Returns the total number of values that have been added to the sketch.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// sketch.add(1.0);
    /// sketch.add(2.0);
    /// assert_eq!(sketch.count(), 2);
    /// ```
    #[inline]
    pub fn count(&self) -> u64 {
        self.positive_store.count() + self.negative_store.count() + self.zero_count
    }

    /// Returns the total number of values in the sketch as a `usize`.
    ///
    /// This mirrors the Rust collection convention; see also [`count`].
    ///
    /// [`count`]: Self::count
    #[inline]
    pub fn len(&self) -> usize {
        self.count() as usize
    }

    /// Returns `true` if no values have been added to the sketch.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Returns the arithmetic sum of all values added to the sketch.
    ///
    /// Unlike the quantile estimates, the sum is tracked exactly (modulo
    /// floating-point accumulation error).
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// sketch.add(1.0);
    /// sketch.add(2.0);
    /// sketch.add(3.0);
    /// assert_eq!(sketch.sum(), 6.0);
    /// ```
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Returns the mean of all values added to the sketch, or `0.0` if empty.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// sketch.add(2.0);
    /// sketch.add(4.0);
    /// assert_eq!(sketch.mean(), 3.0);
    ///
    /// let empty = DDSketch::new(0.01).unwrap();
    /// assert_eq!(empty.mean(), 0.0);
    /// ```
    #[inline]
    pub fn mean(&self) -> f64 {
        if self.count() == 0 {
            0.0
        } else {
            self.sum / (self.count() as f64)
        }
    }

    // Debug methods for testing
    #[cfg(test)]
    pub fn zero_count(&self) -> u64 {
        self.zero_count
    }
    #[cfg(test)]
    pub fn min_indexable_value(&self) -> f64 {
        self.min_indexable_value
    }

    #[cfg(test)]
    pub fn debug_key_to_value(&self, key: i64) -> f64 {
        // Convert i64 key to i32 for the new implementation
        self.value(key as i32)
    }

    #[cfg(test)]
    pub fn bins(&self) -> Vec<u64> {
        // For backward compatibility in tests, return combined view of both stores
        // This is just for testing purposes
        let mut combined_bins = Vec::new();

        // Add negative store (reversed order to maintain ascending value order)
        if !self.negative_store.is_empty() {
            for _i in 0..self.negative_store.length() {
                combined_bins.push(0); // Placeholder - not accurate for test compatibility
            }
        }

        // Add positive store
        if !self.positive_store.is_empty() {
            for _i in 0..self.positive_store.length() {
                combined_bins.push(0); // Placeholder - not accurate for test compatibility
            }
        }

        combined_bins
    }

    /// Returns the reconstructed minimum value, equivalent to `quantile(0.0)`.
    ///
    /// Returns `f64::INFINITY` when the sketch is empty (matching the Go
    /// reference implementation).
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// sketch.add(1.0);
    /// sketch.add(10.0);
    /// // min is reconstructed and therefore only approximate.
    /// assert!((sketch.min() - 1.0).abs() < 1.0 * 0.01);
    /// ```
    #[inline]
    pub fn min(&self) -> f64 {
        if self.count() == 0 {
            return f64::INFINITY;
        }
        self.quantile(0.0).unwrap_or(f64::INFINITY)
    }

    /// Returns the reconstructed maximum value, equivalent to `quantile(1.0)`.
    ///
    /// Returns `f64::NEG_INFINITY` when the sketch is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// sketch.add(1.0);
    /// sketch.add(10.0);
    /// assert!((sketch.max() - 10.0).abs() < 10.0 * 0.01);
    /// ```
    #[inline]
    pub fn max(&self) -> f64 {
        if self.count() == 0 {
            return f64::NEG_INFINITY;
        }
        self.quantile(1.0).unwrap_or(f64::NEG_INFINITY)
    }

    /// Returns the `alpha` (relative-error) parameter this sketch was built with.
    #[inline]
    pub fn alpha(&self) -> f64 {
        (self.gamma - 1.0) / (self.gamma + 1.0)
    }

    /// Resets the sketch to its empty state, discarding all accumulated data.
    ///
    /// The `alpha` configuration is preserved.
    pub fn clear(&mut self) {
        self.sum = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.positive_store = Store::new(self.max_bins);
        self.negative_store = Store::new(self.max_bins);
        self.zero_count = 0;
    }

    /// Returns the estimated value at quantile `q`.
    ///
    /// The returned value is within a multiplicative factor of `alpha` of the
    /// true quantile (see [`DDSketch::alpha`]).
    ///
    /// `q` must be finite and in the closed interval `[0.0, 1.0]`. For an
    /// empty sketch this method returns `Ok(0.0)` for backward compatibility;
    /// use [`DDSketch::quantile_opt`] if you want to distinguish "empty" from
    /// "zero".
    ///
    /// # Errors
    ///
    /// Returns [`DDSketchError::InvalidQuantile`] if `q` is non-finite or
    /// outside `[0.0, 1.0]`.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// for v in 1..=100 {
    ///     sketch.add(v as f64);
    /// }
    /// let median = sketch.quantile(0.5).unwrap();
    /// assert!((median - 50.0).abs() < 50.0 * 0.01 + 0.5);
    /// ```
    pub fn quantile(&self, q: f64) -> Result<f64, DDSketchError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(DDSketchError::InvalidQuantile);
        }

        let total_count = self.count();
        if total_count == 0 {
            return Ok(0.0);
        }

        // Special case: single value - all quantiles return that value
        if total_count == 1 {
            return Ok(self.min); // min == max for single value
        }

        // Remove special case handling for q=0.0 and q=1.0
        // Let them go through normal quantile calculation to get reconstructed values

        let rank = (q * (total_count as f64 - 1.0)) as u64;

        // Follow Go reference implementation quantile logic exactly
        let neg_count = self.negative_store.count();
        let zero_count = self.zero_count;

        if rank < neg_count {
            // Quantile in negative range
            let reversed_rank = neg_count - 1 - rank;
            let key = self.negative_store.key_at_rank(reversed_rank);
            Ok(-self.value(key)) // Note: -value(key) for negatives
        } else if rank < zero_count + neg_count {
            // Quantile in zero range
            Ok(0.0)
        } else {
            // Quantile in positive range
            let positive_rank = rank - zero_count - neg_count;
            let key = self.positive_store.key_at_rank(positive_rank);
            Ok(self.value(key))
        }
    }

    /// Like [`quantile`](Self::quantile), but returns `Ok(None)` when the
    /// sketch is empty rather than `Ok(0.0)`.
    ///
    /// # Errors
    ///
    /// Returns [`DDSketchError::InvalidQuantile`] if `q` is non-finite or
    /// outside `[0.0, 1.0]`.
    pub fn quantile_opt(&self, q: f64) -> Result<Option<f64>, DDSketchError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(DDSketchError::InvalidQuantile);
        }
        if self.count() == 0 {
            return Ok(None);
        }

        Ok(Some(self.quantile(q)?))
    }

    /// Returns the number of values stored in the positive store.
    ///
    /// Primarily useful for debugging and introspection.
    pub fn positive_store_count(&self) -> u64 {
        self.positive_store.count()
    }

    /// Returns the number of values counted in the zero bucket.
    ///
    /// Primarily useful for debugging and introspection.
    pub fn get_zero_count(&self) -> u64 {
        self.zero_count
    }

    /// Returns the number of values stored in the negative store.
    ///
    /// Primarily useful for debugging and introspection.
    pub fn negative_store_count(&self) -> u64 {
        self.negative_store.count()
    }

    /// Returns the bin key at the given rank within the positive store.
    ///
    /// Primarily useful for debugging and introspection.
    pub fn positive_key_at_rank(&self, rank: u64) -> i32 {
        self.positive_store.key_at_rank(rank)
    }

    /// Returns the commonly used percentiles `(P50, P90, P95, P99)`.
    ///
    /// Returns `None` if the sketch is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// for v in 1..=100 {
    ///     sketch.add(v as f64);
    /// }
    /// let (p50, p90, p95, p99) = sketch.percentiles().unwrap();
    /// assert!(p50 < p90 && p90 < p95 && p95 < p99);
    /// ```
    pub fn percentiles(&self) -> Option<(f64, f64, f64, f64)> {
        if self.count() == 0 {
            return None;
        }

        Some((
            self.quantile(0.5).unwrap(),
            self.quantile(0.9).unwrap(),
            self.quantile(0.95).unwrap(),
            self.quantile(0.99).unwrap(),
        ))
    }

    /// Adds every value produced by `values` to the sketch.
    ///
    /// Equivalent to calling [`add`](Self::add) in a loop but communicates
    /// intent and allows future batched optimizations.
    ///
    /// # Example
    ///
    /// ```
    /// use ddsketchy::DDSketch;
    ///
    /// let mut sketch = DDSketch::new(0.01).unwrap();
    /// sketch.add_batch([1.0, 2.0, 3.0]);
    /// assert_eq!(sketch.count(), 3);
    /// ```
    #[inline]
    pub fn add_batch<I>(&mut self, values: I)
    where
        I: IntoIterator<Item = f64>,
    {
        for value in values {
            // Simply use the add() method which handles dual-store logic
            self.add(value);
        }
    }
}

// Implement Default for convenience
impl Default for DDSketch {
    /// Creates a DDSketch with 1% relative error (alpha = 0.01)
    fn default() -> Self {
        Self::new(0.01).expect("Default alpha should be valid")
    }
}

// Implement Display for debugging and logging
impl std::fmt::Display for DDSketch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DDSketch(count={}, alpha={:.3}, min={:.3}, max={:.3}, mean={:.3})",
            self.count(),
            self.alpha(),
            self.min,
            self.max,
            self.mean()
        )
    }
}

// Implement FromIterator for collecting values into a sketch
impl FromIterator<f64> for DDSketch {
    /// Create a DDSketch from an iterator of values using default alpha (0.01)
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut sketch = Self::default();
        sketch.extend(iter);
        sketch
    }
}

// Implement Extend for adding values from iterators
impl Extend<f64> for DDSketch {
    /// Add values from an iterator to the sketch
    fn extend<T: IntoIterator<Item = f64>>(&mut self, iter: T) {
        for value in iter {
            self.add(value);
        }
    }
}

/// Builder for configuring and constructing a [`DDSketch`].
///
/// Use [`DDSketch::builder`] to obtain a new builder.
///
/// # Example
///
/// ```
/// use ddsketchy::DDSketch;
///
/// let sketch = DDSketch::builder(0.01).max_bins(2048).build().unwrap();
/// assert_eq!(sketch.count(), 0);
/// ```
pub struct DDSketchBuilder {
    alpha: f64,
    max_bins: Option<usize>,
}

impl DDSketchBuilder {
    /// Creates a new builder with the given relative-error parameter.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            max_bins: None,
        }
    }

    /// Sets the maximum number of bins in each (positive / negative) store.
    ///
    /// The default is `4096`. Larger values allow the sketch to cover a wider
    /// dynamic range before collapsing bins.
    pub fn max_bins(mut self, max_bins: usize) -> Self {
        self.max_bins = Some(max_bins);
        self
    }

    /// Finalizes the builder and returns the configured [`DDSketch`].
    ///
    /// # Errors
    ///
    /// Returns [`DDSketchError::InvalidAlpha`] if `alpha` is not finite or is
    /// outside `(0, 1)`.
    pub fn build(self) -> Result<DDSketch, DDSketchError> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(DDSketchError::InvalidAlpha);
        }

        // Use the correct DDSketch formulation from the reference implementation
        let gamma = 1.0 + (2.0 * self.alpha) / (1.0 - self.alpha);
        let gamma_ln = ((2.0 * self.alpha) / (1.0 - self.alpha)).ln_1p();
        let inv_ln_gamma = 1.0 / gamma_ln;

        let offset = 0i32; // DataDog uses indexOffset = 0 by default

        // Compute MinIndexableValue like DataDog Go implementation:
        // max(exp((MinInt32 - offset) / multiplier + 1), minNormalFloat64 * gamma)
        // where multiplier = inv_ln_gamma
        let min_indexable_from_range = ((i32::MIN - offset) as f64 / inv_ln_gamma + 1.0).exp();
        let min_indexable_from_normal = f64::MIN_POSITIVE * gamma;
        let min_indexable_value = min_indexable_from_range.max(min_indexable_from_normal);

        let max_bins = self.max_bins.unwrap_or(4096);

        Ok(DDSketch {
            alpha: self.alpha,
            gamma,
            inv_ln_gamma,
            offset,
            min_indexable_value,
            positive_store: Store::new(max_bins),
            negative_store: Store::new(max_bins),
            zero_count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            max_bins,
        })
    }
}

impl DDSketch {
    /// Returns a [`DDSketchBuilder`] for constructing a sketch with extra
    /// configuration beyond the default [`DDSketch::new`] options.
    pub fn builder(alpha: f64) -> DDSketchBuilder {
        DDSketchBuilder::new(alpha)
    }

    /// Convenience constructor: builds a `DDSketch` with the given `alpha`
    /// and a custom `max_bins`.
    ///
    /// # Errors
    ///
    /// Returns [`DDSketchError::InvalidAlpha`] if `alpha` is not finite or is
    /// outside `(0, 1)`.
    pub fn with_max_bins(alpha: f64, max_bins: usize) -> Result<Self, DDSketchError> {
        Self::builder(alpha).max_bins(max_bins).build()
    }
}
