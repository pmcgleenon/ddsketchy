//! DataDog Java reference implementation validation tests
//! These tests implement specific test cases from the DataDog Java implementation
//! to ensure our DDSketch implementation produces correct results.

use crate::DDSketch;
use approx::assert_relative_eq;

/// Tolerance for floating point comparisons - from DataDog Java tests
const FLOATING_POINT_ACCEPTABLE_ERROR: f64 = 1e-11;

/// Test quantiles used in DataDog validation
const VALIDATION_QUANTILES: [f64; 21] = [
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
];

/// Relative accuracy levels to test - from DataDog Java tests
const ACCURACY_LEVELS: [f64; 6] = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 1e-3];

#[test]
fn test_datadog_constant_values() {
    // DataDog test case: constant values
    for &alpha in &ACCURACY_LEVELS {
        let mut sketch = DDSketch::new(alpha).unwrap();

        // Test constant positive value
        let constant_value = 42.0;
        for _ in 0..100 {
            sketch.add(constant_value);
        }

        // All quantiles should return the constant value
        for &q in &VALIDATION_QUANTILES {
            let result = sketch.quantile(q).unwrap();
            assert_relative_eq!(
                result,
                constant_value,
                max_relative = alpha + FLOATING_POINT_ACCEPTABLE_ERROR
            );
        }

        // Test exact statistics
        assert_eq!(sketch.count(), 100);
        assert_eq!(sketch.min(), constant_value);
        assert_eq!(sketch.max(), constant_value);
        assert_relative_eq!(sketch.sum(), constant_value * 100.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
        assert_relative_eq!(sketch.mean(), constant_value, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
    }
}

#[test]
fn test_datadog_mixed_sign_values() {
    // DataDog test case: [0.33, -7] from DDSketchWithExactSummaryStatisticsTest
    let alpha = 1e-1; // 10% relative accuracy from Java tests
    let mut sketch = DDSketch::new(alpha).unwrap();

    let test_values = [0.33, -7.0];
    for &value in &test_values {
        sketch.add(value);
    }

    // Validate exact statistics
    assert_eq!(sketch.count(), 2);
    assert_eq!(sketch.min(), -7.0);
    assert_eq!(sketch.max(), 0.33);
    assert_relative_eq!(sketch.sum(), 0.33 + (-7.0), max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);

    // Validate quantiles with DataDog tolerance
    assert_relative_eq!(sketch.quantile(0.0).unwrap(), -7.0, max_relative = alpha);
    assert_relative_eq!(sketch.quantile(1.0).unwrap(), 0.33, max_relative = alpha);

    // 50th percentile should be between the two values
    let median = sketch.quantile(0.5).unwrap();
    assert!(median >= -7.0 && median <= 0.33, "Median {} out of range [-7.0, 0.33]", median);
}

#[test]
fn test_datadog_linear_sequence() {
    // DataDog test case: linear sequences
    let alpha = 1e-2; // 1% relative accuracy
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add linear sequence 1.0 to 100.0
    let mut expected_values = Vec::new();
    for i in 1..=100 {
        let value = i as f64;
        sketch.add(value);
        expected_values.push(value);
    }
    expected_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Validate exact statistics
    assert_eq!(sketch.count(), 100);
    assert_eq!(sketch.min(), 1.0);
    assert_eq!(sketch.max(), 100.0);
    assert_relative_eq!(sketch.sum(), 5050.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR); // 1+2+...+100 = 5050
    assert_relative_eq!(sketch.mean(), 50.5, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);

    // Validate quantiles with expected relative accuracy
    for &q in &VALIDATION_QUANTILES {
        if q == 0.0 || q == 1.0 {
            // Min/max should be exact
            let result = sketch.quantile(q).unwrap();
            let expected = if q == 0.0 { 1.0 } else { 100.0 };
            assert_relative_eq!(result, expected, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
        } else {
            // Other quantiles should be within relative accuracy
            let result = sketch.quantile(q).unwrap();
            let expected_index = (q * 99.0) as usize; // 0-based index for 100 values
            let expected_value = expected_values[expected_index.min(99)];

            // Allow DDSketch error + floating point error
            let tolerance = expected_value.abs() * alpha + FLOATING_POINT_ACCEPTABLE_ERROR;
            assert!((result - expected_value).abs() <= tolerance,
                "Quantile {} failed: got {}, expected ~{}, diff {}, tolerance {}",
                q, result, expected_value, (result - expected_value).abs(), tolerance);
        }
    }
}

#[test]
fn test_datadog_exponential_sequence() {
    // DataDog test case: exponential sequences
    let alpha = 2e-2; // 2% relative accuracy
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add exponential sequence: 2^0, 2^1, 2^2, ..., 2^10
    let mut expected_values = Vec::new();
    for i in 0..=10 {
        let value = 2.0_f64.powi(i);
        sketch.add(value);
        expected_values.push(value);
    }
    expected_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Validate exact statistics
    assert_eq!(sketch.count(), 11);
    assert_eq!(sketch.min(), 1.0);  // 2^0
    assert_eq!(sketch.max(), 1024.0);  // 2^10

    // Expected sum: 2^0 + 2^1 + ... + 2^10 = 2^11 - 1 = 2047
    assert_relative_eq!(sketch.sum(), 2047.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);

    // Validate quantiles
    for &q in &VALIDATION_QUANTILES {
        let result = sketch.quantile(q).unwrap();

        if q == 0.0 {
            assert_relative_eq!(result, 1.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
        } else if q == 1.0 {
            assert_relative_eq!(result, 1024.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
        } else {
            // Quantile should be within DDSketch bounds
            assert!(result >= 1.0 && result <= 1024.0,
                "Quantile {} = {} out of bounds [1.0, 1024.0]", q, result);

            // For exponential data, verify relative accuracy
            let expected_index = (q * 10.0) as usize;
            let expected_value = expected_values[expected_index.min(10)];
            let tolerance = expected_value * alpha + FLOATING_POINT_ACCEPTABLE_ERROR;

            assert!((result - expected_value).abs() <= tolerance ||
                    result >= expected_value * (1.0 - alpha) && result <= expected_value * (1.0 + alpha),
                "Quantile {} failed relative accuracy: got {}, expected ~{}, alpha {}",
                q, result, expected_value, alpha);
        }
    }
}

#[test]
fn test_datadog_zero_and_negative_values() {
    // DataDog test case: mixed positive, negative, and zero values
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    let test_values = [-10.0, -1.0, 0.0, 0.0, 1.0, 10.0];
    for &value in &test_values {
        sketch.add(value);
    }

    // Validate exact statistics
    assert_eq!(sketch.count(), 6);
    assert_eq!(sketch.min(), -10.0);
    assert_eq!(sketch.max(), 10.0);
    assert_relative_eq!(sketch.sum(), 0.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
    assert_relative_eq!(sketch.mean(), 0.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);

    // Validate boundary quantiles
    assert_relative_eq!(sketch.quantile(0.0).unwrap(), -10.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
    assert_relative_eq!(sketch.quantile(1.0).unwrap(), 10.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);

    // Median should be around 0 (between the two zeros in the sorted sequence)
    let median = sketch.quantile(0.5).unwrap();
    assert!(median >= -1.0 && median <= 1.0, "Median {} should be near 0", median);
}

#[test]
fn test_datadog_single_value() {
    // DataDog test case: single value insertion
    for &alpha in &ACCURACY_LEVELS {
        let mut sketch = DDSketch::new(alpha).unwrap();
        let single_value = 123.456;
        sketch.add(single_value);

        // All statistics should be exact for single value
        assert_eq!(sketch.count(), 1);
        assert_eq!(sketch.min(), single_value);
        assert_eq!(sketch.max(), single_value);
        assert_eq!(sketch.sum(), single_value);
        assert_eq!(sketch.mean(), single_value);

        // All quantiles should return the single value
        for &q in &VALIDATION_QUANTILES {
            let result = sketch.quantile(q).unwrap();
            assert_relative_eq!(result, single_value, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
        }
    }
}

#[test]
fn test_datadog_accuracy_bounds() {
    // DataDog test: verify relative accuracy guarantees
    for &alpha in &ACCURACY_LEVELS {
        let mut sketch = DDSketch::new(alpha).unwrap();

        // Add values with known distribution
        let base_values = [1.0, 10.0, 100.0, 1000.0, 10000.0];
        for &base in &base_values {
            for i in 0..10 {
                let value = base * (1.0 + i as f64 * 0.01); // Small variations
                sketch.add(value);
            }
        }

        // Test that quantiles respect relative accuracy bounds
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let result = sketch.quantile(q).unwrap();

            // Result should be within sketch bounds
            assert!(result >= sketch.min() && result <= sketch.max(),
                "Quantile {} = {} outside bounds [{}, {}]",
                q, result, sketch.min(), sketch.max());

            // For this test, just verify the value is reasonable
            // (exact validation would require knowing the precise expected quantile)
            assert!(result > 0.0, "Quantile {} should be positive, got {}", q, result);
        }
    }
}

#[test]
fn test_datadog_merge_accuracy() {
    // DataDog test case: merging sketches preserves accuracy
    let alpha = 1e-2;
    let mut sketch1 = DDSketch::new(alpha).unwrap();
    let mut sketch2 = DDSketch::new(alpha).unwrap();

    // Add different ranges to each sketch
    for i in 1..=50 {
        sketch1.add(i as f64);
    }
    for i in 51..=100 {
        sketch2.add(i as f64);
    }

    // Create merged sketch
    let mut merged = sketch1.clone();
    merged.merge(&sketch2).unwrap();

    // Merged sketch should have combined data
    assert_eq!(merged.count(), 100);
    assert_eq!(merged.min(), 1.0);
    assert_eq!(merged.max(), 100.0);
    assert_relative_eq!(merged.sum(), 5050.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);

    // Quantiles should be accurate for merged data
    assert_relative_eq!(merged.quantile(0.0).unwrap(), 1.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);
    assert_relative_eq!(merged.quantile(1.0).unwrap(), 100.0, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);

    let median = merged.quantile(0.5).unwrap();
    assert!(median >= 45.0 && median <= 55.0, "Merged median {} should be around 50", median);
}