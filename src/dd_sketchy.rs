//! High-throughput DD Sketch initial implementation with first-phase optimization:
//! replacing division by cached multiplication.
//! No external dependencies.

use std::f64;

/// A simple, optimized DD Sketch implementation.
#[derive(Clone)]
pub struct DDSketch {
    /// Relative error parameter α.
    alpha: f64,
    /// γ = (1+α)/(1-α)
    gamma: f64,
    /// 1 / ln(γ), cached for fast bucket index computation.
    inv_ln_gamma: f64,
    /// Number of bins on each side (positive/negative), zero at center.
    max_bins: usize,
    /// Zero bucket offset index into `bins`.
    offset: usize,
    /// Total count of values.
    count: u64,
    /// Sum of all values (for mean estimation).
    sum: f64,
    /// Pre-allocated bin counts: length = 2*max_bins+1.
    bins: Vec<u64>,
}

impl DDSketch {
    const DEFAULT_MAX_BINS: usize = 2048;

    /// Create a new DD Sketch with relative error `alpha`.
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0,1)");
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        let inv_ln_gamma = 1.0 / gamma.ln();
        let max_bins = Self::DEFAULT_MAX_BINS;
        let size = 2 * max_bins + 1;
        let bins = vec![0u64; size];
        let offset = max_bins;
        Self { alpha, gamma, inv_ln_gamma, max_bins, offset, count: 0, sum: 0.0, bins }
    }

    /// Add a value into the sketch (ignores NaN/Inf).
    #[inline(always)]
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        // Determine bucket index relative to zero.
        let rel_idx = if value == 0.0 {
            0isize
        } else {
            // ceil(log_gamma(|value|))  => ceil(ln(|value|)/ln_gamma)
            let k = (value.abs().ln() * self.inv_ln_gamma).ceil() as isize;
            k
        };
        // Map to absolute bin index, clamped into [0, bins.len()-1]
        let mut idx = if rel_idx > 0 {
            self.offset + (rel_idx as usize)
        } else if rel_idx < 0 {
            self.offset.saturating_sub((-rel_idx) as usize)
        } else {
            self.offset
        };
        if idx >= self.bins.len() {
            idx = self.bins.len() - 1;
        }
        self.bins[idx] += 1;
        self.count += 1;
        self.sum += value;
    }

    /// Merge another sketch into this one (in-place).
    pub fn merge(&mut self, other: &Self) {
        assert!((self.alpha - other.alpha).abs() < f64::EPSILON, "alpha must match");
        assert!(self.max_bins == other.max_bins, "bin count must match");
        for (a, b) in self.bins.iter_mut().zip(other.bins.iter()) {
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
        for (i, &c) in self.bins.iter().enumerate() {
            if rem <= c {
                return self.bin_to_value(i);
            }
            rem -= c;
        }
        // fallback to max bin
        self.bin_to_value(self.bins.len() - 1)
    }

    /// Convert a bin index back to a representative value (bucket midpoint).
    fn bin_to_value(&self, idx: usize) -> f64 {
        let rel = idx as isize - self.offset as isize;
        if rel == 0 {
            return 0.0;
        }
        let k = rel.abs() as f64;
        let v = self.gamma.powf(k - 1.0) * (1.0 + self.alpha);
        if rel < 0 { -v } else { v }
    }
}

