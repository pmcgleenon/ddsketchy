//! High-throughput DD Sketch initial implementation with first-phase optimization:
//! replacing division by cached multiplication.
//! No external dependencies.

use core::f64;

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
    
    /// Precomputed values for faster bin_to_value
    inv_ln_gamma: f64,
    
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
            inv_ln_gamma,
            bins: AlignedBuckets { bins: [0; 4096] },
            alpha,
            max_bins: 4096,
            offset: 2048,
        })
    }

    /// Add a value into the sketch (ignores NaN/Inf).
    /// Uses unsafe pointer writes to avoid bounds checks on increment.
    #[inline(always)]
    pub fn add(&mut self, value: f64) {
        // Early check for non-finite values
        if !value.is_finite() {
            return;
        }

        // Parallel computation of conditions
        let abs_value = value.abs();
        let is_zero = value == Self::ZERO;
        let is_positive = value >= Self::ZERO;
        
        // Independent computations
        let log_value = if !is_zero { abs_value.ln() } else { Self::ZERO };
        let scaled_log = log_value * self.inv_ln_gamma;
        // zero goes in the zero-bin; everything else must map to at least bucket 1
        let rel_idx = if is_zero {
            0
        } else {
            let raw = scaled_log.ceil() as isize;
            if raw < 1 { 1 } else { raw }
        };
        
        // Update statistics
        self.count += 1;
        self.sum += value;
        
        // Handle zero case early
        if is_zero {
            self.bins.bins[self.offset] += 1;
            return;
        }
        
        // Compute bucket index with bounds checking
        let idx = if is_positive {
            self.offset.saturating_add(rel_idx as usize)
        } else {
            self.offset.saturating_sub(rel_idx as usize)
        };
        
        // Update bucket count if within bounds
        if idx < self.bins.bins.len() {
            self.bins.bins[idx] += 1;
        } else {
            // If out of bounds, increment the last bin
            self.bins.bins[if value >= Self::ZERO { self.bins.bins.len() - 1 } else { 0 }] += 1;
        }
    }

    /// Merge another sketch into this one (in-place).
    /// Returns an error if the sketches have different alpha values or bin counts.
    #[inline(always)]
    pub fn merge(&mut self, other: &Self) -> Result<(), DDSketchError> {
        if (self.alpha - other.alpha).abs() >= f64::EPSILON {
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
    pub fn quantile(&self, q: f64) -> Result<f64, DDSketchError> {
        if q < 0.0 || q > 1.0 {
            return Err(DDSketchError::InvalidQuantile);
        }
        if self.count == 0 {
            return Ok(0.0);
        }

        // new: use floor((count-1)*q)+1 so that
        //   q=0.33 with count=4 → floor(3*0.33)=0 → rem=1 → 1st item
        let n = self.count;
        let idx0 = (((n - 1) as f64) * q).floor() as u64;
        let mut rem = idx0.saturating_add(1);

        // scan bins in ascending order
        for (i, &c) in self.bins.bins.iter().enumerate() {
            if rem <= c {
                return Ok(self.bin_to_value(i));
            }
            rem -= c;
        }

        // fallback to max bin
        Ok(self.bin_to_value(self.bins.bins.len() - 1))
    }

    /// Convert a bin index back to a representative value (bucket midpoint).
    /// Optimized version with improved branch prediction.
    #[inline(always)]
    fn bin_to_value(&self, idx: usize) -> f64 {
        // Early return for zero case (most common)
        if idx == self.offset {
            return Self::ZERO;
        }
        
        // Calculate relative index
        let rel = idx as isize - self.offset as isize;
        if rel == 0 {
            return Self::ZERO;
        }
        
        // Compute value using gamma^k
        let gamma = (1.0 + self.alpha) / (1.0 - self.alpha);
        let k = rel.abs() as f64;
        let value = gamma.powf(k - 1.0) * (1.0 + self.alpha);
        
        // Apply sign
        if rel < 0 { -value } else { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add_zero() {
        let mut dd = DDSketch::new(0.01).unwrap();
        dd.add(0.0);
        assert_eq!(dd.count(), 1);
        assert_eq!(dd.sum(), 0.0);
    }

    #[test]
    fn test_quartiles() {
        let mut dd = DDSketch::new(0.01).unwrap();

        // Initialize sketch with {1.0, 2.0, 3.0, 4.0}
        for i in 1..5 {
            dd.add(i as f64);
        }

        // We expect the following mappings from quantile to value:
        // [0,0.33]: 1.0, (0.34,0.66]: 2.0, (0.67,0.99]: 3.0, (0.99, 1.0]: 4.0
        let test_cases = vec![
            (0.0, 1.0),
            (0.25, 1.0),
            (0.33, 1.0),
            (0.34, 2.0),
            (0.5, 2.0),
            (0.66, 2.0),
            (0.67, 3.0),
            (0.75, 3.0),
            (0.99, 3.0),
            (1.0, 4.0),
        ];

        for (q, val) in test_cases {
            assert_relative_eq!(dd.quantile(q).unwrap(), val, max_relative = 0.01);
        }
    }

    #[test]
    fn test_neg_quartiles() {
        let mut dd = DDSketch::new(0.01).unwrap();

        // Initialize sketch with {-1.0, -2.0, -3.0, -4.0}
        for i in 1..5 {
            dd.add(-i as f64);
        }

        let test_cases = vec![
            (0.0, -4.0),
            (0.25, -4.0),
            (0.5, -3.0),
            (0.75, -2.0),
            (1.0, -1.0),
        ];

        for (q, val) in test_cases {
            assert_relative_eq!(dd.quantile(q).unwrap(), val, max_relative = 0.01);
        }
    }

    #[test]
    fn test_simple_quantile() {
        let mut dd = DDSketch::new(0.01).unwrap();

        for i in 1..101 {
            dd.add(i as f64);
        }

        assert_eq!(dd.quantile(0.95).unwrap().ceil(), 95.0);

        assert!(dd.quantile(-1.01).is_err());
        assert!(dd.quantile(1.01).is_err());
    }

    #[test]
    fn test_empty_sketch() {
        let dd = DDSketch::new(0.01).unwrap();

        assert_eq!(dd.quantile(0.98).unwrap(), 0.0);
        assert_eq!(dd.count(), 0);
        assert_eq!(dd.sum(), 0.0);

        assert!(dd.quantile(1.01).is_err());
    }

    #[test]
    fn test_basic_histogram_data() {
        let values = &[
            0.754225035,
            0.752900282,
            0.752812246,
            0.752602367,
            0.754310155,
            0.753525981,
            0.752981082,
            0.752715536,
            0.751667941,
            0.755079054,
            0.753528150,
            0.755188464,
            0.752508723,
            0.750064549,
            0.753960428,
            0.751139298,
            0.752523560,
            0.753253428,
            0.753498342,
            0.751858358,
            0.752104636,
            0.753841300,
            0.754467374,
            0.753814334,
            0.750881719,
            0.753182556,
            0.752576884,
            0.753945708,
            0.753571911,
            0.752314573,
            0.752586651,
        ];

        let mut dd = DDSketch::new(0.01).unwrap();

        for value in values {
            dd.add(*value);
        }

        assert_eq!(dd.count(), 31);
        assert_relative_eq!(dd.sum(), 23.343630625000003, max_relative = 0.01);

        assert!(dd.quantile(0.25).is_ok());
        assert!(dd.quantile(0.5).is_ok());
        assert!(dd.quantile(0.75).is_ok());
    }

    #[test]
    fn test_invalid_alpha() {
        assert!(matches!(DDSketch::new(0.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(DDSketch::new(1.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(DDSketch::new(-1.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(DDSketch::new(2.0), Err(DDSketchError::InvalidAlpha)));
    }
}

