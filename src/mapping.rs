/// Shared DDSketch key/value mapping functions
/// Map a value to its corresponding DDSketch key
/// Uses the standard DDSketch formula: ceil(ln(value) / ln(gamma))
///
#[inline]
pub fn value_to_key(value: f64, gamma_ln: f64) -> i64 {
    if value == 0.0 {
        return 0;
    }
    let log_gamma = value.abs().ln() / gamma_ln;
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
pub fn value_to_key_i32(value: f64, gamma_ln: f64) -> i32 {
    value_to_key(value, gamma_ln) as i32
}

/// Helper function for the main DDSketch implementation that needs i32 keys
#[inline]
pub fn key_to_value_i32(key: i32, gamma: f64, gamma_ln: f64) -> f64 {
    key_to_value(key as i64, gamma, gamma_ln)
}
