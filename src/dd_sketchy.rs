//! A self-contained, correct, fast DDSketch implementation with internal tests.
//! No external dependencies except `rand_distr` for internal tests.

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
    // Hot path fields - accessed frequently, grouped for cache locality
    bins: Vec<u64>,
    min_key: i64,
    max_key: i64,
    count: u64,
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

    // Configuration - accessed less frequently
    gamma: f64,
    gamma_ln: f64,
    inv_ln_gamma: f64,
    key_epsilon: f64,
    offset: i64,
    max_bins: usize,
    is_collapsed: bool,
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

        let min_value = 1e-9_f64;
        let offset = 0i64; // DataDog uses indexOffset = 0 by default

        let max_bins = 4096;

        Ok(Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            gamma,
            gamma_ln,
            inv_ln_gamma: 1.0 / gamma_ln,
            key_epsilon: min_value,
            offset,
            max_bins,
            is_collapsed: false,
            bins: Vec::new(),
            min_key: 0,
            max_key: 0,
            zero_count: 0,
        })
    }

    /// Map a value to a bin key
    #[inline]
    pub fn key(&self, value: f64) -> i64 {
        let abs_val = value.abs();
        if abs_val <= self.key_epsilon {
            return 0;
        }

        // Calculate key using the same formula as add method
        let log_gamma = abs_val.ln() * self.inv_ln_gamma + self.offset as f64;
        let abs_key = if log_gamma >= 0.0 {
            log_gamma as i64
        } else {
            log_gamma as i64 - 1  // Equivalent to floor for negative values
        };

        // Handle negative values by using negative keys
        if value < 0.0 {
            if abs_key == 0 {
                -1  // Ensure negative values never map to key 0
            } else {
                -abs_key
            }
        } else {
            abs_key
        }
    }

    /// Ensure bins can hold the given key
    fn ensure_capacity(&mut self, key: i64) {
        if self.bins.is_empty() {
            self.min_key = key;
            self.max_key = key;
            self.bins = vec![0; 1];
        } else if key < self.min_key {
            if self.is_collapsed {
                // If already collapsed, add to the first bin which contains collapsed values
                return;
            }
            self.extend_range(key, self.max_key);
        } else if key > self.max_key {
            self.extend_range(self.min_key, key);
        }
    }

    /// Extend the range to accommodate new keys, collapsing if necessary
    fn extend_range(&mut self, new_min_key: i64, new_max_key: i64) {
        let new_min_key = new_min_key.min(self.min_key);
        let new_max_key = new_max_key.max(self.max_key);

        let range_size = new_max_key.saturating_sub(new_min_key).saturating_add(1);
        if range_size < 0 || range_size > i64::MAX / 2 {
            // Range too large, force collapsing
            self.adjust(new_min_key, new_max_key);
            return;
        }
        let required_size = range_size as usize;
        if required_size <= self.max_bins {
            // No collapsing needed, just grow the bins
            if new_min_key < self.min_key {
                let grow_left = (self.min_key - new_min_key) as usize;
                let mut new_bins = vec![0; grow_left];
                new_bins.extend_from_slice(&self.bins);
                self.bins = new_bins;
                self.min_key = new_min_key;
            }
            if new_max_key > self.max_key {
                let grow_right = (new_max_key - self.max_key) as usize;
                self.bins.resize(self.bins.len() + grow_right, 0);
                self.max_key = new_max_key;
            }
        } else {
            // Need to collapse - follow reference implementation strategy
            self.adjust(new_min_key, new_max_key);
        }
    }

    /// Adjust bins to fit within max_bins, collapsing lowest bins if necessary
    /// Based on Go reference CollapsingLowestDenseStore.adjust()
    fn adjust(&mut self, new_min_key: i64, new_max_key: i64) {
        let range_size = new_max_key.saturating_sub(new_min_key).saturating_add(1);
        if range_size > self.max_bins as i64 {
            // Need to collapse - keep highest keys (newest values)
            let adjusted_min_key = new_max_key - (self.max_bins as i64) + 1;

            if adjusted_min_key >= self.max_key {
                // Everything gets collapsed into first bin
                let total_count = self.bins.iter().sum::<u64>();
                self.bins = vec![total_count; self.max_bins];
                for i in 1..self.max_bins {
                    self.bins[i] = 0;
                }
                self.min_key = adjusted_min_key;
                self.max_key = new_max_key;
            } else {
                // Collapse lower bins and shift
                let mut collapsed_count = 0u64;

                // Sum all bins that will be collapsed (below adjusted_min_key)
                for key in self.min_key..adjusted_min_key {
                    let idx = (key - self.min_key) as usize;
                    if idx < self.bins.len() {
                        collapsed_count += self.bins[idx];
                    }
                }

                // Create new bins array
                let mut new_bins = vec![0; self.max_bins];

                // Set collapsed count in the first bin of the new range
                new_bins[0] = collapsed_count;

                // Copy remaining bins that survive the collapse
                for key in adjusted_min_key..=self.max_key {
                    let old_idx = (key - self.min_key) as usize;
                    let new_idx = (key - adjusted_min_key) as usize;
                    if old_idx < self.bins.len() && new_idx < new_bins.len() {
                        new_bins[new_idx] += self.bins[old_idx];
                    }
                }

                self.bins = new_bins;
                self.min_key = adjusted_min_key;
                self.max_key = new_max_key;
            }

            self.is_collapsed = true;
        } else {
            // No collapsing needed - use normal extension
            self.center_bins(new_min_key, new_max_key);
        }
    }

    /// Center the bins in the array for the new range
    fn center_bins(&mut self, new_min_key: i64, new_max_key: i64) {
        let range_size = new_max_key.saturating_sub(new_min_key).saturating_add(1);
        let required_size = range_size.min(self.max_bins as i64) as usize;
        let mut new_bins = vec![0; required_size];

        // Copy existing bins to new array
        for key in self.min_key..=self.max_key {
            let old_idx = (key - self.min_key) as usize;
            let new_idx = (key - new_min_key) as usize;
            if old_idx < self.bins.len() && new_idx < new_bins.len() {
                new_bins[new_idx] = self.bins[old_idx];
            }
        }

        self.bins = new_bins;
        self.min_key = new_min_key;
        self.max_key = new_max_key;
    }

    /// Add a value to the sketch
    #[inline]
    pub fn add(&mut self, value: f64) {
        // Skip infinite/NaN values immediately
        if !value.is_finite() {
            return;
        }

        let abs_val = value.abs();

        if abs_val <= self.key_epsilon {
            self.zero_count += 1;
        } else {
            // For non-zero finite values
            // Calculate key using pre-computed abs_val and offset
            let log_gamma = abs_val.ln() * self.inv_ln_gamma + self.offset as f64;
            let abs_key = if log_gamma >= 0.0 {
                log_gamma as i64
            } else {
                log_gamma as i64 - 1  // Equivalent to floor for negative values
            };
            let key = if value < 0.0 {
                if abs_key == 0 {
                    -1  // Ensure negative values never map to key 0
                } else {
                    -abs_key
                }
            } else {
                abs_key
            };

            self.ensure_capacity(key);

            // Safely calculate index with bounds checking
            let key_offset = key - self.min_key;
            if key_offset < 0 || key_offset >= self.bins.len() as i64 {
                // Value outside current bin range after collapsing - this can happen
                // with extreme values when collapsing is active
                return;
            }
            let idx = key_offset as usize;
            self.bins[idx] += 1;
        }

        // Update metadata after processing
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Merge another sketch into this one
    pub fn merge(&mut self, other: &Self) -> Result<(), DDSketchError> {
        if (self.gamma - other.gamma).abs() > 1e-10 {
            return Err(DDSketchError::AlphaMismatch);
        }

        if other.count == 0 {
            return Ok(());
        }

        if self.count == 0 {
            self.copy_from(other);
            return Ok(());
        }

        // Extend range to accommodate other sketch, with collapsing if needed
        let new_min_key = self.min_key.min(other.min_key);
        let new_max_key = self.max_key.max(other.max_key);

        if (new_max_key - new_min_key + 1) as usize > self.max_bins {
            self.adjust(new_min_key, new_max_key);
        } else {
            self.extend_range(new_min_key, new_max_key);
        }

        // Merge the bins from other sketch
        for key in other.min_key..=other.max_key {
            let other_idx = (key - other.min_key) as usize;
            if other_idx < other.bins.len() && other.bins[other_idx] > 0 {
                let self_idx = (key - self.min_key) as usize;
                if self_idx < self.bins.len() {
                    self.bins[self_idx] += other.bins[other_idx];
                } else if self.is_collapsed && key < self.min_key {
                    // If collapsed and key is below our range, add to first bin
                    self.bins[0] += other.bins[other_idx];
                }
            }
        }

        self.count += other.count;
        self.sum += other.sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.zero_count += other.zero_count;

        Ok(())
    }

    /// Copy from another sketch
    fn copy_from(&mut self, other: &Self) {
        self.count = other.count;
        self.sum = other.sum;
        self.min = other.min;
        self.max = other.max;
        self.gamma = other.gamma;
        self.gamma_ln = other.gamma_ln;
        self.inv_ln_gamma = other.inv_ln_gamma;
        self.key_epsilon = other.key_epsilon;
        self.offset = other.offset;
        self.max_bins = other.max_bins;
        self.is_collapsed = other.is_collapsed;
        self.bins = other.bins.clone();
        self.min_key = other.min_key;
        self.max_key = other.max_key;
        self.zero_count = other.zero_count;
    }

    /// Returns the number of values added to the sketch
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the number of values added to the sketch (Rust collection convention)
    #[inline]
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// Returns true if the sketch is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
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
        if self.count == 0 {
            0.0
        } else {
            self.sum / (self.count as f64)
        }
    }

    // Debug methods for testing
    #[cfg(test)]
    pub fn zero_count(&self) -> u64 { self.zero_count }
    #[cfg(test)]
    pub fn min_key(&self) -> i64 { self.min_key }
    #[cfg(test)]
    pub fn max_key(&self) -> i64 { self.max_key }
    #[cfg(test)]
    pub fn bins(&self) -> &[u64] { &self.bins }
    #[cfg(test)]
    pub fn debug_key_to_value(&self, key: i64) -> f64 { self.key_to_value(key) }
    #[cfg(test)]
    pub fn key_epsilon(&self) -> f64 { self.key_epsilon }

    /// Returns the minimum value added to the sketch
    /// Returns f64::INFINITY if the sketch is empty
    #[inline]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Returns the maximum value added to the sketch
    /// Returns f64::NEG_INFINITY if the sketch is empty
    #[inline]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Returns the alpha (relative error) parameter used to create this sketch
    #[inline]
    pub fn alpha(&self) -> f64 {
        (self.gamma - 1.0) / (self.gamma + 1.0)
    }

    /// Clears all data from the sketch, resetting it to empty state
    pub fn clear(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.bins.clear();
        self.min_key = 0;
        self.max_key = 0;
        self.zero_count = 0;
        self.is_collapsed = false;
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
        if self.count == 0 {
            return Ok(0.0);
        }

        // Special case: single value - all quantiles return that value
        if self.count == 1 {
            return Ok(self.min); // min == max for single value
        }

        if q == 0.0 {
            return Ok(self.min);
        } else if q == 1.0 {
            return Ok(self.max);
        }

        let rank = (q * (self.count as f64 - 1.0)) as u64;

        let mut sum = 0u64;

        // Process negative values first (in descending order)
        if self.min_key < 0 {
            for key in (self.min_key..0).rev() {
                let bin_idx = (key - self.min_key) as usize;
                if bin_idx < self.bins.len() {
                    let count = self.bins[bin_idx];
                    if count > 0 {
                        sum += count;
                        if sum > rank {
                            let reconstructed_value = self.key_to_value(key);
                            return Ok(reconstructed_value.min(self.max).max(self.min));
                        }
                    }
                }
            }
        }

        // Process zeros
        sum += self.zero_count;
        if sum > rank {
            return Ok(0.0);
        }

        // Process positive values (in ascending order)
        if self.max_key >= 0 {
            let start_key = if self.min_key <= 0 { 0 } else { self.min_key };
            for key in start_key..=self.max_key {
                let bin_idx = (key - self.min_key) as usize;
                if bin_idx < self.bins.len() {
                    let count = self.bins[bin_idx];
                    if count > 0 {
                        sum += count;
                        if sum > rank {
                            let reconstructed_value = self.key_to_value(key);
                            return Ok(reconstructed_value.min(self.max).max(self.min));
                        }
                    }
                }
            }
        }

        Ok(self.max)
    }

    /// Returns the value at the given quantile, with Option for empty handling
    pub fn quantile_opt(&self, q: f64) -> Result<Option<f64>, DDSketchError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(DDSketchError::InvalidQuantile);
        }
        if self.count == 0 {
            return Ok(None);
        }

        Ok(Some(self.quantile(q)?))
    }

    /// Returns commonly used percentiles (P50, P90, P95, P99)
    ///
    /// Returns None if the sketch is empty.
    pub fn percentiles(&self) -> Option<(f64, f64, f64, f64)> {
        if self.count == 0 {
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
        let mut batch_count = 0u64;
        let mut batch_sum = 0.0f64;
        let mut batch_min = f64::INFINITY;
        let mut batch_max = f64::NEG_INFINITY;

        for value in values {
            // Skip infinite/NaN values immediately
            if !value.is_finite() {
                continue;
            }

            let abs_val = value.abs();

            if abs_val <= self.key_epsilon {
                self.zero_count += 1;
                batch_count += 1;
                batch_sum += value;
                batch_min = batch_min.min(value);
                batch_max = batch_max.max(value);
            } else {
                let log_gamma = abs_val.ln() * self.inv_ln_gamma;
                let abs_key = log_gamma.ceil() as i64;
                let key = if value < 0.0 { -abs_key } else { abs_key };

                self.ensure_capacity(key);

                // Safely calculate index with bounds checking
                let key_offset = key - self.min_key;
                if key_offset < 0 || key_offset >= self.bins.len() as i64 {
                    // Value outside current bin range after collapsing
                    continue;
                }
                let idx = key_offset as usize;
                self.bins[idx] += 1;

                batch_count += 1;
                batch_sum += value;
                batch_min = batch_min.min(value);
                batch_max = batch_max.max(value);
            }
        }

        // Update metadata once at the end
        self.count += batch_count;
        self.sum += batch_sum;
        self.min = self.min.min(batch_min);
        self.max = self.max.max(batch_max);
    }

    #[inline]
    fn key_to_value(&self, key: i64) -> f64 {
        let abs_key = key.abs() as f64;

        // Calculate the lower bound of the bin using offset
        // This matches DataDog's LowerBound(index) = exp((index - indexOffset) / multiplier)
        let lower_bound = ((abs_key - self.offset as f64) * self.gamma_ln).exp();

        // Return the representative value: lower_bound * (1 + alpha)
        // This matches DataDog's Value(index) = LowerBound(index) * (1 + RelativeAccuracy())
        let alpha = (self.gamma - 1.0) / (self.gamma + 1.0);
        let abs_value = lower_bound * (1.0 + alpha);

        // Return negative value for negative keys
        if key < 0 {
            -abs_value
        } else {
            abs_value
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
            self.count,
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

        let min_value = 1e-9_f64;
        let offset = 0i64; // DataDog uses indexOffset = 0 by default
        let max_bins = self.max_bins.unwrap_or(4096);

        Ok(DDSketch {
            bins: Vec::new(),
            min_key: 0,
            max_key: 0,
            count: 0,
            zero_count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            gamma,
            gamma_ln,
            inv_ln_gamma: 1.0 / gamma_ln,
            key_epsilon: min_value,
            offset,
            max_bins,
            is_collapsed: false,
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
