//! High-throughput DD Sketch initial implementation with first-phase optimization:
//! replacing division by cached multiplication.
//! No external dependencies.

use std::f64;

/// A cache-line aligned array of bucket counts
#[derive(Clone)]
#[repr(align(64))]
struct AlignedBuckets {
    bins: [u64; 4097]
}

/// A simple, optimized DD Sketch implementation.
#[derive(Clone)]
#[repr(align(64))]  // Align entire struct to cache line boundary
pub struct DDSketch {
    /// Frequently accessed fields first
    count: u64,
    sum: f64,
    
    /// Precomputed values for faster bin_to_value
    inv_ln_gamma: f64,
    ln_gamma: f64,
    alpha_plus_one: f64,
    
    /// Then the bins array
    bins: AlignedBuckets,
    
    /// Less frequently accessed fields
    alpha: f64,
    gamma: f64,
    max_bins: usize,
    offset: usize,
}

impl DDSketch {
    /// Precomputed constants
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const NEG_ONE: f64 = -1.0;
    
    /// Create a new DD Sketch with relative error `alpha`.
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0,1)");
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        let ln_gamma = gamma.ln();
        let inv_ln_gamma = 1.0 / ln_gamma;
        let alpha_plus_one = 1.0 + alpha;
        
        Self {
            count: 0,
            sum: 0.0,
            inv_ln_gamma,
            ln_gamma,
            alpha_plus_one,
            bins: AlignedBuckets { bins: [0; 4097] },
            alpha,
            gamma,
            max_bins: 4096,
            offset: 2048,
        }
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
        let rel_idx = if is_zero { 0 } else { scaled_log.ceil() as isize };
        
        // Update statistics
        self.count += 1;
        self.sum += value;
        
        // Handle zero case early
        if is_zero {
            self.bins.bins[self.offset] += 1;
            return;
        }
        
        // Compute bucket index
        let idx = if is_positive {
            self.offset + rel_idx as usize
        } else {
            self.offset - rel_idx as usize
        };
        
        // Update bucket count
        if idx < self.bins.bins.len() {
            self.bins.bins[idx] += 1;
        }
    }

    /// Merge another sketch into this one (in-place).
    pub fn merge(&mut self, other: &Self) {
        assert!((self.alpha - other.alpha).abs() < f64::EPSILON, "alpha must match");
        assert!(self.max_bins == other.max_bins, "bin count must match");
        for (a, b) in self.bins.bins.iter_mut().zip(other.bins.bins.iter()) {
            *a += *b;
        }
        self.count += other.count;
        self.sum += other.sum;
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
    pub fn quantile(&self, q: f64) -> f64 {
        assert!(q >= 0.0 && q <= 1.0, "q must be between 0 and 1");
        if self.count == 0 {
            return 0.0;
        }
        let mut rem = (q * (self.count as f64)).ceil() as u64;
        for (i, &c) in self.bins.bins.iter().enumerate() {
            if rem <= c {
                return self.bin_to_value(i);
            }
            rem -= c;
        }
        // fallback to max bin
        self.bin_to_value(self.bins.bins.len() - 1)
    }

    /// Convert a bin index back to a representative value (bucket midpoint).
    /// Optimized version with reduced branches and memory pressure.
    #[inline(always)]
    fn bin_to_value(&self, idx: usize) -> f64 {
        // Calculate relative index and its sign in parallel
        let rel = idx as isize - self.offset as isize;
        let k = rel.abs() as f64;
        
        // Compute exp(ln_gamma * (k-1)) * alpha_plus_one
        let k_minus_one = k - Self::ONE;
        let exp_term = (k_minus_one * self.ln_gamma).exp();
        let abs_result = exp_term * self.alpha_plus_one;
        
        // Use arithmetic for branchless operations
        let is_zero = (rel == 0) as i64 as f64;
        let is_negative = (rel < 0) as i64 as f64;
        
        // Combine results using arithmetic instead of branches
        let sign = 1.0 - 2.0 * is_negative;
        abs_result * sign * (1.0 - is_zero)
    }
}

