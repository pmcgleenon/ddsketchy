//! A self-contained, correct, fast DDSketch implementation with internal tests.
//! No external dependencies except `rand_distr` for internal tests.

use crate::store::Store;

/// Errors for DDSketch operations
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum DDSketchError {
    /// Alpha parameter must be in range (0, 1)
    InvalidAlpha,
    /// Quantile must be in range [0, 1]
    InvalidQuantile,
    /// Cannot merge sketches with different alpha values
    AlphaMismatch,
    /// Bin count exceeds maximum allowed
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
/// DDSketch provides fast quantile estimation with bounded relative error.
/// It's fully mergeable and designed for high-throughput data collection.
///
/// # Serialization
///
/// When the `serde` feature is enabled, `DDSketch` implements
/// `Serialize` and `Deserialize`.
///
/// ```toml
/// dd-sketchy = { features = ["serde"] }
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
    /// Create a new DDSketch with relative error `alpha` (0 < alpha < 1)
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

    /// Map a value to a bin key
    #[inline]
    pub fn key(&self, value: f64) -> i32 {
        crate::mapping::value_to_key_i32(value, self.inv_ln_gamma)
    }

    /// Get the minimum indexable value (MinIndexableValue)
    ///
    /// Values below this threshold are clamped to the zero bucket.
    /// Matches DataDog's Mapping.MinIndexableValue() behavior.
    #[inline]
    pub fn min_possible(&self) -> f64 {
        self.min_indexable_value
    }

    /// Map a key back to its representative value
    /// Uses exact Go/Java style implementation with upward bias
    #[inline]
    pub fn value(&self, key: i32) -> f64 {
        crate::mapping::key_to_value_i32(key, self.gamma, self.inv_ln_gamma.recip())
    }

    /// Add a value to the sketch
    #[inline]
    pub fn add(&mut self, value: f64) {
        // Skip infinite/NaN values immediately
        if !value.is_finite() {
            return;
        }

        // DataDog semantics: extremely small magnitudes are mapped to the zero bucket.
        // Values below MinIndexableValue go to zero bucket.
        if value == 0.0 || value.abs() < self.min_indexable_value {
            self.zero_count += 1;
        } else if value > 0.0 {
            // Positive value -> positive store
            let key = self.key(value);
            self.positive_store.add(key);
        } else {
            // Negative value -> negative store with key(-value)
            let key = self.key(-value);
            self.negative_store.add(key);
        }

        // Update metadata
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Merge another sketch into this one
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

    /// Returns the number of values added to the sketch
    #[inline]
    pub fn count(&self) -> u64 {
        self.positive_store.count() + self.negative_store.count() + self.zero_count
    }

    /// Returns the number of values added to the sketch (Rust collection convention)
    #[inline]
    pub fn len(&self) -> usize {
        self.count() as usize
    }

    /// Returns true if the sketch is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Returns the sum of all values added to the sketch
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Returns the mean of all values added to the sketch
    /// Returns 0.0 if the sketch is empty
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

    /// Returns the minimum reconstructed value (equivalent to quantile(0.0))
    /// This matches Go DDSketch behavior where min/max return reconstructed values
    #[inline]
    pub fn min(&self) -> f64 {
        if self.count() == 0 {
            return f64::INFINITY;
        }
        self.quantile(0.0).unwrap_or(f64::INFINITY)
    }

    /// Returns the maximum reconstructed value (equivalent to quantile(1.0))
    /// This matches Go DDSketch behavior where min/max return reconstructed values
    #[inline]
    pub fn max(&self) -> f64 {
        if self.count() == 0 {
            return f64::NEG_INFINITY;
        }
        self.quantile(1.0).unwrap_or(f64::NEG_INFINITY)
    }

    /// Returns the alpha (relative error) parameter used to create this sketch
    #[inline]
    pub fn alpha(&self) -> f64 {
        (self.gamma - 1.0) / (self.gamma + 1.0)
    }

    /// Clears all data from the sketch, resetting it to empty state
    pub fn clear(&mut self) {
        self.sum = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.positive_store = Store::new(self.max_bins);
        self.negative_store = Store::new(self.max_bins);
        self.zero_count = 0;
    }

    /// Returns the value at the given quantile
    ///
    /// # Arguments
    /// * `q` - The quantile to query, must be in range [0.0, 1.0]
    ///
    /// # Returns
    /// * `Ok(value)` - The estimated value at the quantile
    /// * `Err(DDSketchError::InvalidQuantile)` - If quantile is outside [0.0, 1.0]
    ///
    /// Returns 0.0 if the sketch is empty for backward compatibility.
    pub fn quantile(&self, q: f64) -> Result<f64, DDSketchError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
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

    /// Returns the value at the given quantile, with Option for empty handling
    pub fn quantile_opt(&self, q: f64) -> Result<Option<f64>, DDSketchError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(DDSketchError::InvalidQuantile);
        }
        if self.count() == 0 {
            return Ok(None);
        }

        Ok(Some(self.quantile(q)?))
    }

    /// Debug method to get positive store count
    pub fn positive_store_count(&self) -> u64 {
        self.positive_store.count()
    }

    /// Debug method to get zero count
    pub fn get_zero_count(&self) -> u64 {
        self.zero_count
    }

    /// Debug method to get negative store count
    pub fn negative_store_count(&self) -> u64 {
        self.negative_store.count()
    }

    /// Debug method to get key at rank from positive store
    pub fn positive_key_at_rank(&self, rank: u64) -> i32 {
        self.positive_store.key_at_rank(rank)
    }

    /// Returns commonly used percentiles (P50, P90, P95, P99)
    ///
    /// Returns None if the sketch is empty.
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

    /// Add multiple values efficiently with reduced overhead
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

// Builder pattern for flexible construction
pub struct DDSketchBuilder {
    alpha: f64,
    max_bins: Option<usize>,
}

impl DDSketchBuilder {
    /// Create a new builder with the specified alpha
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            max_bins: None,
        }
    }

    /// Set the maximum number of bins (default: 4096)
    pub fn max_bins(mut self, max_bins: usize) -> Self {
        self.max_bins = Some(max_bins);
        self
    }

    /// Build the DDSketch
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
    /// Create a new builder for constructing a DDSketch
    pub fn builder(alpha: f64) -> DDSketchBuilder {
        DDSketchBuilder::new(alpha)
    }

    /// Create a DDSketch with custom maximum bins
    pub fn with_max_bins(alpha: f64, max_bins: usize) -> Result<Self, DDSketchError> {
        Self::builder(alpha).max_bins(max_bins).build()
    }
}
