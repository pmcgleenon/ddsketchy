/// Shared DDSketch key/value mapping functions
/// Map a value to its corresponding DDSketch key
/// Uses the optimized formula: ceil(ln(value) * inv_ln_gamma)
/// where inv_ln_gamma = 1.0 / ln(gamma)
#[inline]
pub fn value_to_key(value: f64, inv_gamma_ln: f64) -> i64 {
    if value == 0.0 {
        return 0;
    }
    let log_gamma = value.abs().ln() * inv_gamma_ln;
    log_gamma.ceil() as i64
}

/// Map a DDSketch key back to its representative value
/// Uses the standard DDSketch formula: γ^k * (2 / (1 + γ))
///
/// This is equivalent to: exp(k * ln(γ)) * (2 / (1 + γ))
/// Both formulas produce the same result but the exponential form
/// avoids potential overflow with very large keys.
#[inline]
pub fn key_to_value(key: i64, gamma: f64, gamma_ln: f64) -> f64 {
    // Use the exponential form: exp(key * ln(gamma)) * (2 / (1 + gamma))
    // This handles negative keys correctly and avoids special case for key=0
    let key_f64 = key as f64;
    (key_f64 * gamma_ln).exp() * (2.0 / (1.0 + gamma))
}

/// Helper function for the main DDSketch implementation that needs i32 keys
#[inline]
pub fn value_to_key_i32(value: f64, inv_gamma_ln: f64) -> i32 {
    value_to_key(value, inv_gamma_ln) as i32
}

/// Helper function for the main DDSketch implementation that needs i32 keys
#[inline]
pub fn key_to_value_i32(key: i32, gamma: f64, gamma_ln: f64) -> f64 {
    key_to_value(key as i64, gamma, gamma_ln)
}

/// Fixed exponent that tightens gamma to compensate for [`fast_ln`]'s own
/// approximation error, so the combined (approximation + quantization) error
/// a sketch built with the adjusted gamma sees still stays within the
/// originally requested `alpha`. Derived analytically from this specific
/// cubic polynomial's coefficients by Datadog; see
/// <https://github.com/DataDog/sketches-go/blob/master/ddsketch/mapping/cubically_interpolated_mapping.go>.
/// Apply as `gamma_ln * CUBIC_GAMMA_EXPONENT` wherever `fast_ln` is used in
/// place of `ln()` for bucket assignment — both the forward mapping and its
/// inverse must use the *same* adjusted gamma to remain exact inverses of
/// each other (see `DDSketch::new` and `DDSketch::value`).
pub const CUBIC_GAMMA_EXPONENT: f64 = 10.0 * std::f64::consts::LN_2 / 7.0;

/// Cubic-interpolated approximation of `ln(value)`, avoiding a transcendental
/// `f64::ln()` call on the insert hot path.
///
/// Approximates `log2(value)` directly from the IEEE-754 bit layout
/// (exponent + a cubic polynomial over the mantissa's fractional part), then
/// converts to natural log. Ported from Datadog's cubic mapping
/// (<https://github.com/DataDog/sketches-go/blob/master/ddsketch/mapping/cubically_interpolated_mapping.go>,
/// also used by <https://github.com/tomtomwombat/quantile-sketch>).
///
/// This approximation is **not** tight enough to use as a drop-in for
/// `ln()` on its own — empirically its relative error can reach a few
/// percent (see `tests::fast_ln_error_is_bounded`), which is too large
/// against a typical DDSketch alpha of 0.1%-5%. It must be paired with
/// [`CUBIC_GAMMA_EXPONENT`]'s gamma adjustment at the sketch level to bring
/// the *combined* error back within budget; see `DDSketch::new`.
///
/// # Preconditions
///
/// Only valid for finite, normal, positive `value` (`value >=
/// f64::MIN_POSITIVE`); not enforced by an assertion. Callers must ensure
/// this — `DDSketch::add` already filters non-finite values and routes
/// sub-`min_indexable_value` magnitudes to the zero bucket before this is
/// called.
#[inline]
pub fn fast_ln(value: f64) -> f64 {
    const MANTISSA_MASK: u64 = (1_u64 << 52) - 1;
    const ONE_BITS: u64 = 1023_u64 << 52;
    // Coefficients of the cubic interpolation of log2's fractional part.
    const A: f64 = 6.0 / 35.0;
    const B: f64 = -3.0 / 5.0;
    const C: f64 = 10.0 / 7.0;

    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let s = f64::from_bits((bits & MANTISSA_MASK) | ONE_BITS) - 1.0;
    let log2 = ((A * s + B) * s + C) * s + exponent as f64;
    log2 * std::f64::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_ln_error_is_bounded() {
        // Document (not hide) the raw approximation's own error: it's a few
        // percent on its own, which is why `CUBIC_GAMMA_EXPONENT` exists to
        // compensate at the sketch level (see `fast_ln`'s doc comment and
        // `ddsketchy_test::...` for the end-to-end alpha guarantee this
        // combination actually achieves).
        let mut max_rel_error = 0.0_f64;
        for exponent in -300..300 {
            let base = 2.0_f64.powi(exponent);
            for i in 0..1000 {
                let value = base * (1.0 + i as f64 / 1000.0);
                if !value.is_finite() || value <= 0.0 {
                    continue;
                }
                let exact = value.ln();
                let approx = fast_ln(value);
                let rel_error = (approx - exact).abs() / exact.abs().max(1e-300);
                max_rel_error = max_rel_error.max(rel_error);
            }
        }
        assert!(
            max_rel_error < 0.05,
            "fast_ln relative error out of the expected (compensated-for) range: {max_rel_error}"
        );
    }

    #[test]
    fn fast_ln_is_monotonic() {
        // `Store::add` relies on the key sequence being non-decreasing in
        // value; a non-monotonic approximation would silently scramble
        // bucket ordering.
        let mut prev = f64::NEG_INFINITY;
        for i in 1..1_000_000 {
            let value = i as f64 * 0.0001;
            let approx = fast_ln(value);
            assert!(
                approx >= prev,
                "fast_ln must be non-decreasing: value={value}"
            );
            prev = approx;
        }
    }
}
