//! DD Sketch implementation:
//!
//! No external dependencies.


/// Errors that can occur during DDSketch operations
#[derive(Debug, PartialEq)]
pub enum DDSketchError {
    /// The alpha values of two sketches don't match
    AlphaMismatch,
    /// The bin counts of two sketches don't match
    BinCountMismatch,
    /// The quantile provided is not in the range [0,1]
    InvalidQuantile,
    /// The alpha value provided is not in the range (0,1)
    InvalidAlpha,
}

/// A cache-line aligned array of bucket counts
#[derive(Clone, Debug, PartialEq)]
#[repr(align(64))]
struct AlignedBuckets {
    bins: [u64; 4096]
}

/// A simple, optimized DD Sketch implementation.
#[derive(Clone, Debug, PartialEq)]
#[repr(align(64))]  // Align entire struct to cache line boundary
pub struct DDSketch {
    /// Frequently accessed fields first
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
    
    /// Precomputed values for faster bin_to_value
    inv_ln_gamma: f64,
    gamma: f64,
    
    /// Then the bins array
    bins: AlignedBuckets,
    
    /// Less frequently accessed fields
    alpha: f64,
    max_bins: usize,
    offset: usize,
    non_empty_buckets: usize,
}

impl DDSketch {
    /// Precomputed constants
    const ZERO: f64 = 0.0;
    
    /// Create a new DD Sketch with relative error `alpha`.
    pub fn new(alpha: f64) -> Result<Self, DDSketchError> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(DDSketchError::InvalidAlpha);
        }
        // Correct gamma calculation to match reference implementation
        let gamma = 1.0 + (2.0 * alpha) / (1.0 - alpha);
        let gamma_ln = gamma.ln();

        Ok(Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            inv_ln_gamma: 1.0 / gamma_ln,
            gamma,
            bins: AlignedBuckets { bins: [0; 4096] },
            alpha,
            max_bins: 4096,
            offset: 2048,
            non_empty_buckets: 0,
        })
    }

    /// Find the two smallest non-empty buckets and merge them
    fn collapse_smallest_buckets(&mut self) {
        let mut smallest_idx = None;
        let mut second_smallest_idx = None;
        let mut smallest_count = u64::MAX;
        let mut second_smallest_count = u64::MAX;

        // Find two smallest non-empty buckets
        for (i, &count) in self.bins.bins.iter().enumerate() {
            if count == 0 {
                continue;
            }
            if count < smallest_count {
                second_smallest_count = smallest_count;
                second_smallest_idx = smallest_idx;
                smallest_count = count;
                smallest_idx = Some(i);
            } else if count < second_smallest_count {
                second_smallest_count = count;
                second_smallest_idx = Some(i);
            }
        }

        // If we found two buckets to merge
        if let (Some(i), Some(j)) = (smallest_idx, second_smallest_idx) {
            // Merge into the higher index bucket
            let (lower_idx, higher_idx) = if i < j { (i, j) } else { (j, i) };
            self.bins.bins[higher_idx] += self.bins.bins[lower_idx];
            self.bins.bins[lower_idx] = 0;
            // Decrement non_empty_buckets since we merged two buckets
            self.non_empty_buckets -= 1;
        }
    }

    /// Add a value into the sketch (ignores NaN/Inf).
    #[inline(always)]
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        // Update statistics
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Handle zero case early
        if value == Self::ZERO {
            if self.bins.bins[self.offset] == 0 {
                self.non_empty_buckets += 1;
            }
            self.bins.bins[self.offset] += 1;
            return;
        }

        let abs_value = value.abs();
        let idx = if abs_value == Self::ZERO {
            self.offset
        } else {
            // Use log_gamma(abs(value)) for bin index
            let log_gamma = abs_value.ln() * self.inv_ln_gamma;
            let rel = log_gamma.ceil() as isize;
            let idx = if value >= Self::ZERO {
                self.offset as isize + rel
            } else {
                self.offset as isize - rel
            };
            if idx < 0 {
                0
            } else if (idx as usize) >= self.bins.bins.len() {
                self.bins.bins.len() - 1
            } else {
                idx as usize
            }
        };

        // Update bucket count and non_empty_buckets
        if self.bins.bins[idx] == 0 {
            self.non_empty_buckets += 1;
        }
        self.bins.bins[idx] += 1;

        // Check if we need to collapse buckets
        if self.count & 0xFF == 0 && self.non_empty_buckets > self.max_bins {
            self.collapse_smallest_buckets();
        }
    }

    /// Merge another sketch into this one (in-place).
    /// Returns an error if the sketches have different alpha values or bin counts.
    #[inline(always)]
    pub fn merge(&mut self, other: &Self) -> Result<(), DDSketchError> {
        // Check alpha compatibility with reasonable tolerance
        let alpha_diff = (self.alpha - other.alpha).abs();
        if alpha_diff > 1e-10 {
            return Err(DDSketchError::AlphaMismatch);
        }
        if self.max_bins != other.max_bins {
            return Err(DDSketchError::BinCountMismatch);
        }
        
        // Update non_empty_buckets count
        for (a, b) in self.bins.bins.iter_mut().zip(other.bins.bins.iter()) {
            if *b > 0 {
                if *a == 0 {
                    self.non_empty_buckets += 1;
                }
                *a += *b;
            }
        }
        self.count += other.count;
        self.sum += other.sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        Ok(())
    }

    /// Number of values inserted.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sum of all values.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Mean of the observations.
    pub fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / (self.count as f64) }
    }

    /// Estimate the q-th quantile (q in [0,1]).
    #[inline(always)]
    pub fn quantile(&self, q: f64) -> Result<f64, DDSketchError> {
        if q < 0.0 || q > 1.0 {
            return Err(DDSketchError::InvalidQuantile);
        }
        if self.count == 0 {
            return Ok(0.0);
        }

        // Handle extreme quantiles using exact min/max
        if q == 0.0 {
            return Ok(self.min);
        }
        if q == 1.0 {
            return Ok(self.max);
        }

        // Calculate rank as in sketches-ddsketch:
        // rank = q * (count - 1)
        let rank = (q * (self.count as f64 - 1.0)) as u64;
        let mut sum = 0;

        // Simple linear scan
        for (i, &c) in self.bins.bins.iter().enumerate() {
            sum += c;
            if sum > rank {
                // If this is the first non-empty bin, return min
                if i == 0 || self.bins.bins[..i].iter().all(|&x| x == 0) {
                    return Ok(self.min);
                }
                // If this is the last non-empty bin, return max
                if i == self.bins.bins.len() - 1 || self.bins.bins[i+1..].iter().all(|&x| x == 0) {
                    return Ok(self.max);
                }
                // For non-extreme quantiles, use the bin value directly
                return Ok(self.bin_to_value(i));
            }
        }

        // If we get here, return max
        Ok(self.max)
    }

    /// Convert a bin index back to a representative value (bucket midpoint).
    #[inline(always)]
    fn bin_to_value(&self, idx: usize) -> f64 {
        // Early return for zero case
        if idx == self.offset {
            return Self::ZERO;
        }

        // Calculate relative index
        let rel = idx as isize - self.offset as isize;
        // Calculate the value using the gamma formula: value = sign * gamma^k * (2/(1+gamma))
        let k = rel.abs() as f64;
        let v = self.gamma.powf(k) * (2.0 / (1.0 + self.gamma));
        if rel < 0 { -v } else { v }
    }
}

