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
}

impl DDSketch {
    /// Precomputed constants
    const ZERO: f64 = 0.0;
    
    /// Create a new DD Sketch with relative error `alpha`.
    pub fn new(alpha: f64) -> Result<Self, DDSketchError> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(DDSketchError::InvalidAlpha);
        }
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        let ln_gamma = gamma.ln();
        let inv_ln_gamma = 1.0 / ln_gamma;

        Ok(Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            inv_ln_gamma,
            gamma,
            bins: AlignedBuckets { bins: [0; 4096] },
            alpha,
            max_bins: 4096,
            offset: 2048,
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
        }
    }

    /// Count non-empty buckets
    fn count_non_empty_buckets(&self) -> usize {
        self.bins.bins.iter().filter(|&&count| count > 0).count()
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
            self.bins.bins[self.offset] += 1;
            return;
        }

        // Compute bucket index in one pass
        let abs_value = value.abs();
        let idx = if abs_value == Self::ZERO {
            self.offset
        } else {
            let scaled_log = abs_value.ln() * self.inv_ln_gamma;
            let raw = scaled_log.ceil() as isize;
            let rel = if raw < 1 { 1 } else { raw };
            
            if value >= Self::ZERO {
                self.offset.saturating_add(rel as usize)
            } else {
                self.offset.saturating_sub(rel as usize)
            }
        };

        // Update bucket count
        if idx < self.bins.bins.len() {
            self.bins.bins[idx] += 1;
        } else {
            // If out of bounds, increment the last bin
            self.bins.bins[if value >= Self::ZERO { self.bins.bins.len() - 1 } else { 0 }] += 1;
        }

        // Check if we need to collapse buckets
        // Only check every N inserts to amortize cost
        if self.count & 0xFF == 0 && self.count_non_empty_buckets() > self.max_bins {
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
        
        for (a, b) in self.bins.bins.iter_mut().zip(other.bins.bins.iter()) {
            *a += *b;
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
        // rank = floor(q * count + 0.5)
        let rank = ((q * (self.count as f64)) + 0.5).floor() as u64;
        let mut sum = 0;

        // Simple linear scan
        for (i, &c) in self.bins.bins.iter().enumerate() {
            sum += c;
            if sum >= rank {
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
        // weighted midpoint = γ^(k−1)·(1+α)
        let k = rel.abs() as f64;
        let v = self.gamma.powf(k - 1.0) * (1.0 + self.alpha);
        if rel < 0 { -v } else { v }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    const RELATIVE_ERROR: f64 = 0.01;
    const TEST_VALUES: &[f64] = &[
        0.754225035, 0.752900282, 0.752812246, 0.752602367, 0.754310155,
        0.753525981, 0.752981082, 0.752715536, 0.751667941, 0.755079054,
        0.753528150, 0.755188464, 0.752508723, 0.750064549, 0.753960428,
        0.751139298, 0.752523560, 0.753253428, 0.753498342, 0.751858358,
        0.752104636, 0.753841300, 0.754467374, 0.753814334, 0.750881719,
        0.753182556, 0.752576884, 0.753945708, 0.753571911, 0.752314573,
        0.752586651,
    ];

    #[test]
    fn test_add_zero() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
        dd.add(0.0);
        assert_eq!(dd.count(), 1);
        assert_eq!(dd.sum(), 0.0);
        assert_eq!(dd.quantile(0.5).unwrap(), 0.0);
    }

    #[test]
    fn test_empty_sketch() {
        let dd = DDSketch::new(RELATIVE_ERROR).unwrap();
        assert_eq!(dd.quantile(0.98).unwrap(), 0.0);
        assert_eq!(dd.count(), 0);
        assert_eq!(dd.sum(), 0.0);
        assert!(dd.quantile(1.01).is_err());
    }

    #[test]
    fn test_invalid_alpha() {
        assert!(matches!(DDSketch::new(0.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(DDSketch::new(1.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(DDSketch::new(-1.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(DDSketch::new(2.0), Err(DDSketchError::InvalidAlpha)));
    }

    #[test]
    fn test_basic_histogram_data() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
        for &value in TEST_VALUES {
            dd.add(value);
        }
        assert_eq!(dd.count(), TEST_VALUES.len() as u64);
        assert_relative_eq!(dd.sum(), TEST_VALUES.iter().sum(), max_relative = RELATIVE_ERROR);
        
        // Test various quantiles
        assert!(dd.quantile(0.25).is_ok());
        assert!(dd.quantile(0.5).is_ok());
        assert!(dd.quantile(0.75).is_ok());
    }

    #[test]
    fn test_constant_values() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
        let constant = 42.0;
        for _ in 0..100 {
            dd.add(constant);
        }
        for q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_relative_eq!(dd.quantile(*q).unwrap(), constant, max_relative = RELATIVE_ERROR);
        }
    }

    #[test]
    fn test_linear_distribution() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
        let values: Vec<f64> = (0..100).map(|x| x as f64).collect();
        for &v in &values {
            dd.add(v);
        }

        // Test various quantiles with appropriate error bounds
        let test_cases = vec![
            (0.0, 0.0),     // min
            (0.1, 9.0),     // p10
            (0.25, 24.0),   // p25
            (0.5, 49.5),    // median
            (0.75, 74.0),   // p75
            (0.9, 89.0),    // p90
            (1.0, 99.0),    // max
        ];

        for &(q, expected) in &test_cases {
            let actual = dd.quantile(q).unwrap();
            
            // For min/max, expect exact values
            if q == 0.0 || q == 1.0 {
                assert_relative_eq!(actual, expected, max_relative = RELATIVE_ERROR);
                continue;
            }

            // For other quantiles, allow broader bounds
            let min_expected = expected * 0.98; // -2%
            let max_expected = expected * 1.02; // +2%
            assert!(
                actual >= min_expected && actual <= max_expected,
                "q={}, got={}, expected=[{}, {}]",
                q, actual, min_expected, max_expected
            );
        }
    }

    #[test]
    fn test_normal_distribution() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let mean = 100.0;
        let std_dev = 10.0;
        
        // Generate normal distribution
        for _ in 0..1000 {
            let v = rng.gen::<f64>() * std_dev + mean;
            dd.add(v);
        }
        
        // The median should be close to the mean
        assert_relative_eq!(dd.quantile(0.5).unwrap(), mean, max_relative = 0.1);
    }

    #[test]
    fn test_quartiles() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();

        // Initialize sketch with {1.0, 2.0, 3.0, 4.0}
        for i in 1..5 {
            dd.add(i as f64);
        }

        // Test exact quantile values
        let test_cases = vec![
            (0.0, 1.0),   // min
            (0.25, 1.0),  // first quartile
            (0.5, 2.0),   // median
            (0.75, 3.0),  // third quartile
            (1.0, 4.0),   // max
        ];

        for (q, expected) in test_cases {
            let actual = dd.quantile(q).unwrap();
            assert_relative_eq!(actual, expected, max_relative = RELATIVE_ERROR,
                epsilon = RELATIVE_ERROR,
                max_relative = RELATIVE_ERROR);
        }
    }

    #[test]
    fn test_neg_quartiles() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();

        // Initialize sketch with {-1.0, -2.0, -3.0, -4.0}
        for i in 1..5 {
            dd.add(-(i as f64));
        }

        let test_cases = vec![
            (0.0, -4.0),   // min
            (0.25, -4.0),  // first quartile
            (0.5, -3.0),   // median
            (0.75, -2.0),  // third quartile
            (1.0, -1.0),   // max
        ];

        for (q, expected) in test_cases {
            let actual = dd.quantile(q).unwrap();
            assert_relative_eq!(actual, expected, max_relative = RELATIVE_ERROR,
                epsilon = RELATIVE_ERROR,
                max_relative = RELATIVE_ERROR);
        }
    }

    #[test]
    fn test_simple_quantile() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();

        for i in 1..101 {
            dd.add(i as f64);
        }

        assert_eq!(dd.quantile(0.95).unwrap().ceil(), 95.0);
        assert!(dd.quantile(-1.01).is_err());
        assert!(dd.quantile(1.01).is_err());
    }

    #[test]
    fn test_extreme_values() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
        
        // Add some extreme values
        dd.add(1e-100);
        dd.add(1e100);
        dd.add(0.0);
        
        assert_relative_eq!(dd.quantile(0.0).unwrap(), 1e-100, max_relative = RELATIVE_ERROR);
        assert_relative_eq!(dd.quantile(1.0).unwrap(), 1e100, max_relative = RELATIVE_ERROR);
    }

    #[test]
    fn test_mixed_distribution() {
        let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
        
        // Mix of positive, negative, and zero values
        let values = vec![-10.0, -1.0, 0.0, 0.0, 1.0, 10.0];
        for &v in &values {
            dd.add(v);
        }
        
        assert_relative_eq!(dd.quantile(0.0).unwrap(), -10.0, max_relative = RELATIVE_ERROR);
        assert_relative_eq!(dd.quantile(0.5).unwrap(), 0.0, max_relative = RELATIVE_ERROR);
        assert_relative_eq!(dd.quantile(1.0).unwrap(), 10.0, max_relative = RELATIVE_ERROR);
    }

    #[test]
    fn test_merge_operations() {
        let mut d1 = DDSketch::new(RELATIVE_ERROR).unwrap();
        let mut d2 = DDSketch::new(RELATIVE_ERROR).unwrap();

        // Simple constant values like in the sketches-ddsketch example
        d1.add(1.0);
        d2.add(2.0);
        d2.add(2.0);

        // Merge d2 into d1
        d1.merge(&d2).unwrap();

        // Basic count check like in sketches-ddsketch
        assert_eq!(d1.count(), 3);

        // Additional thorough checks
        assert_relative_eq!(d1.sum(), 5.0, max_relative = RELATIVE_ERROR);
        assert_relative_eq!(d1.quantile(0.0).unwrap(), 1.0, max_relative = RELATIVE_ERROR);
        assert_relative_eq!(d1.quantile(1.0).unwrap(), 2.0, max_relative = RELATIVE_ERROR);
        
        // For the median, we expect it to be 2.0 since we have {1.0, 2.0, 2.0}
        let median = d1.quantile(0.5).unwrap();
        assert!(
            (median - 2.0).abs() <= 2.0 * RELATIVE_ERROR,
            "median {} not within {}% of expected 2.0",
            median,
            RELATIVE_ERROR * 100.0
        );
    }

    #[test]
    fn test_merge_error_cases() {
        let mut d1 = DDSketch::new(0.01).unwrap();
        let mut d2 = DDSketch::new(0.02).unwrap(); // Different alpha

        d1.add(1.0);
        d2.add(2.0);

        // Test alpha mismatch
        assert!(matches!(d1.merge(&d2), Err(DDSketchError::AlphaMismatch)));

        // Test bin count mismatch (would require modifying the struct to have different max_bins)
        // This is more of a theoretical test since we can't easily create sketches with different bin counts
    }

    #[test]
    fn test_merge_empty_sketches() {
        let mut d1 = DDSketch::new(RELATIVE_ERROR).unwrap();
        let d2 = DDSketch::new(RELATIVE_ERROR).unwrap();

        // Merge empty sketch
        d1.merge(&d2).unwrap();
        assert_eq!(d1.count(), 0);
        assert_eq!(d1.sum(), 0.0);
    }

    #[test]
    fn test_merge_with_extreme_values() {
        let mut d1 = DDSketch::new(RELATIVE_ERROR).unwrap();
        let mut d2 = DDSketch::new(RELATIVE_ERROR).unwrap();

        d1.add(1e-100);
        d2.add(1e100);

        d1.merge(&d2).unwrap();

        assert_eq!(d1.count(), 2);
        assert_relative_eq!(d1.quantile(0.0).unwrap(), 1e-100, max_relative = RELATIVE_ERROR);
        assert_relative_eq!(d1.quantile(1.0).unwrap(), 1e100, max_relative = RELATIVE_ERROR);
    }
}

