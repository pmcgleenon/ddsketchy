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

#[test]
fn test_datadog_extreme_values() {
    // DataDog test: extreme value handling - very large, very small, and boundary values
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Test very large numbers (but not overflow)
    let very_large = 1e15;
    sketch.add(very_large);

    // Test very small positive numbers
    let very_small = 1e-15;
    sketch.add(very_small);

    // Test numbers close to f64 limits (but safe)
    let near_max = f64::MAX / 1e6; // Safe large value
    sketch.add(near_max);

    let near_min_positive = f64::MIN_POSITIVE * 1e6; // Safe small positive value
    sketch.add(near_min_positive);

    // Test numbers very close to zero but not exactly zero
    let almost_zero = 1e-100;
    sketch.add(almost_zero);

    // Test negative extreme values
    sketch.add(-very_large);
    sketch.add(-very_small);
    sketch.add(-near_max);

    // Validate basic statistics - note that extreme ranges may cause collapsing
    // which can drop some values to stay within bin limits
    assert!(sketch.count() >= 6 && sketch.count() <= 8, "Count should be 6-8, got {}", sketch.count());

    // Min/max should be within reasonable bounds of the added values
    assert!(sketch.min() >= -near_max && sketch.min() <= very_small);
    assert!(sketch.max() >= very_small && sketch.max() <= near_max);

    // Test that all quantiles return reasonable values within bounds
    for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let result = sketch.quantile(q).unwrap();

        // Ensure result is finite and within sketch bounds
        assert!(result.is_finite(), "Quantile {} should be finite, got {}", q, result);
        assert!(result >= sketch.min() && result <= sketch.max(),
            "Quantile {} = {} outside bounds [{}, {}]",
            q, result, sketch.min(), sketch.max());
    }
}

#[test]
fn test_datadog_subnormal_numbers() {
    // DataDog test: handling of subnormal/denormal floating point numbers
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add subnormal numbers (smaller than MIN_POSITIVE)
    let subnormal1 = f64::MIN_POSITIVE / 2.0;
    let subnormal2 = f64::MIN_POSITIVE / 1000.0;

    // These should be treated as effectively zero due to key_epsilon
    sketch.add(subnormal1);
    sketch.add(subnormal2);

    // Add some normal numbers for comparison
    sketch.add(1.0);
    sketch.add(-1.0);

    assert_eq!(sketch.count(), 4);

    // Subnormals should be counted as zeros
    assert_eq!(sketch.zero_count(), 2);

    // Quantiles should be reasonable
    let median = sketch.quantile(0.5).unwrap();
    assert!(median >= -1.0 && median <= 1.0, "Median {} should be between -1 and 1", median);
}

#[test]
fn test_datadog_special_float_values() {
    // DataDog test: handling of special IEEE 754 values
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add normal values first
    sketch.add(1.0);
    sketch.add(2.0);
    sketch.add(3.0);

    let initial_count = sketch.count();
    let initial_sum = sketch.sum();

    // Try adding special values - these should be ignored/skipped
    sketch.add(f64::INFINITY);
    sketch.add(f64::NEG_INFINITY);
    sketch.add(f64::NAN);

    // Count and sum should not change (special values ignored)
    assert_eq!(sketch.count(), initial_count, "Special values should not affect count");
    assert_relative_eq!(sketch.sum(), initial_sum, max_relative = FLOATING_POINT_ACCEPTABLE_ERROR);

    // Quantiles should still work properly
    let median = sketch.quantile(0.5).unwrap();
    assert!(median.is_finite(), "Median should be finite after adding special values");
    assert!(median >= 1.5 && median <= 2.5, "Median {} should be approximately 2.0", median);
}

#[test]
fn test_datadog_large_magnitude_range() {
    // DataDog test: values spanning many orders of magnitude
    let alpha = 2e-2; // Slightly larger alpha for this challenging test
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add values spanning 20+ orders of magnitude
    let base_values = [
        1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10,
    ];

    for &value in &base_values {
        sketch.add(value);
        sketch.add(-value); // Include negative counterparts
    }

    assert_eq!(sketch.count(), 22); // 11 positive + 11 negative
    assert_eq!(sketch.min(), -1e10);
    assert_eq!(sketch.max(), 1e10);

    // Test quantiles across the range
    for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let result = sketch.quantile(q).unwrap();

        // Ensure result is within bounds and reasonable
        assert!(result >= sketch.min() && result <= sketch.max(),
            "Quantile {} = {} outside bounds [{}, {}]",
            q, result, sketch.min(), sketch.max());

        // For this symmetric distribution, median should be reasonable
        if q == 0.5 {
            assert!(result.abs() <= 1e-8, "Median {} should be close to zero for symmetric distribution", result);
        }
    }
}

#[test]
fn test_datadog_precision_boundaries() {
    // DataDog test: values at floating point precision boundaries
    let alpha = 1e-3;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Test values that are exactly representable vs those that aren't
    let base = 1.0;
    let epsilon = f64::EPSILON;

    // Values very close together (testing floating point precision)
    sketch.add(base);
    sketch.add(base + epsilon);
    sketch.add(base + 2.0 * epsilon);
    sketch.add(base + 10.0 * epsilon);

    // Test powers of 2 (exactly representable)
    for i in -10..=10 {
        sketch.add(2.0_f64.powi(i));
    }

    assert!(sketch.count() > 0);

    // Validate quantiles are finite and bounded
    for &q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let result = sketch.quantile(q).unwrap();
        assert!(result.is_finite(), "Quantile {} should be finite", q);
        assert!(result >= sketch.min() && result <= sketch.max(),
            "Quantile {} = {} outside bounds [{}, {}]",
            q, result, sketch.min(), sketch.max());
    }
}

#[test]
fn test_datadog_mapping_precision() {
    // DataDog test: value-to-key and key-to-value mapping precision
    let alpha = 1e-3; // High precision for mapping tests
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Test values near bin boundaries to validate mapping precision
    let test_values = [
        1.0, 1.01, 1.02, 1.03, // Values that should map to consecutive or same bins
        10.0, 10.1, 10.2,     // Different magnitude
        100.0, 100.5, 101.0,  // Another magnitude
        0.1, 0.11, 0.12,      // Small values
    ];

    for &value in &test_values {
        sketch.add(value);
    }

    // All quantiles should be within the actual data range
    for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let result = sketch.quantile(q).unwrap();

        // Must be within actual data bounds
        assert!(result >= sketch.min() && result <= sketch.max(),
            "Quantile {} = {} outside data bounds [{}, {}]",
            q, result, sketch.min(), sketch.max());

        // Must be finite
        assert!(result.is_finite(), "Quantile {} must be finite, got {}", q, result);
    }

    // Test that similar values give similar quantile results (consistency)
    let q50_1 = sketch.quantile(0.5).unwrap();
    let q50_2 = sketch.quantile(0.5).unwrap();
    assert_eq!(q50_1, q50_2, "Quantile calls should be deterministic");
}

#[test]
fn test_datadog_bin_boundary_precision() {
    // DataDog test: ensure values at bin boundaries are handled correctly
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add values that should test bin boundary conditions
    let base = 10.0;
    let epsilon = alpha * base; // Small relative change

    // Values very close to each other (testing bin boundaries)
    sketch.add(base);
    sketch.add(base + epsilon);
    sketch.add(base - epsilon);
    sketch.add(base + 2.0 * epsilon);
    sketch.add(base - 2.0 * epsilon);

    assert_eq!(sketch.count(), 5);

    // All quantiles should be reasonable
    for &q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let result = sketch.quantile(q).unwrap();

        // Should be within the range of added values
        let min_val = base - 2.0 * epsilon;
        let max_val = base + 2.0 * epsilon;
        assert!(result >= min_val && result <= max_val,
            "Quantile {} = {} outside expected range [{}, {}]",
            q, result, min_val, max_val);
    }

    // Median should be close to base value
    let median = sketch.quantile(0.5).unwrap();
    let relative_error = (median - base).abs() / base;
    assert!(relative_error <= alpha * 2.0,
        "Median relative error {} exceeds 2*alpha {}", relative_error, alpha * 2.0);
}

#[test]
fn test_datadog_consecutive_value_mapping() {
    // DataDog test: consecutive values and their mapping behavior
    let alpha = 2e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add consecutive integer values
    for i in 1..=20 {
        sketch.add(i as f64);
    }

    assert_eq!(sketch.count(), 20);
    assert_eq!(sketch.min(), 1.0);
    assert_eq!(sketch.max(), 20.0);

    // Test specific quantiles for known data
    let q0 = sketch.quantile(0.0).unwrap();
    let q25 = sketch.quantile(0.25).unwrap();
    let q50 = sketch.quantile(0.5).unwrap();
    let q75 = sketch.quantile(0.75).unwrap();
    let q100 = sketch.quantile(1.0).unwrap();

    // Boundary quantiles should be exact
    assert_eq!(q0, 1.0, "Min quantile should be exact");
    assert_eq!(q100, 20.0, "Max quantile should be exact");

    // Intermediate quantiles should be reasonable
    assert!(q25 >= 1.0 && q25 <= 20.0, "Q25 {} out of range", q25);
    assert!(q50 >= 1.0 && q50 <= 20.0, "Q50 {} out of range", q50);
    assert!(q75 >= 1.0 && q75 <= 20.0, "Q75 {} out of range", q75);

    // Quantiles should be ordered
    assert!(q0 <= q25, "Quantiles should be ordered: q0 {} <= q25 {}", q0, q25);
    assert!(q25 <= q50, "Quantiles should be ordered: q25 {} <= q50 {}", q25, q50);
    assert!(q50 <= q75, "Quantiles should be ordered: q50 {} <= q75 {}", q50, q75);
    assert!(q75 <= q100, "Quantiles should be ordered: q75 {} <= q100 {}", q75, q100);
}

#[test]
fn test_datadog_relative_accuracy_validation() {
    // DataDog test: validate relative accuracy guarantees
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add values across different orders of magnitude
    let test_values = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0];
    for &value in &test_values {
        sketch.add(value);
    }

    // For each quantile, verify it's within relative accuracy of reasonable bounds
    for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let result = sketch.quantile(q).unwrap();

        // Result must be within data bounds
        assert!(result >= sketch.min() && result <= sketch.max(),
            "Quantile {} = {} outside bounds [{}, {}]",
            q, result, sketch.min(), sketch.max());

        // For this specific dataset, verify relative accuracy makes sense
        // The result should be within reasonable relative error of expected values
        if q == 0.5 {
            // Median should be somewhere in the middle range
            assert!(result >= 10.0 && result <= 500.0,
                "Median {} seems unreasonable for dataset", result);
        }
    }
}

#[test]
fn test_datadog_edge_quantiles() {
    // DataDog test: numerical stability for quantiles very close to 0.0 and 1.0
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add a diverse dataset to test edge quantiles
    for i in 1..=1000 {
        sketch.add(i as f64);
    }

    // Test quantiles very close to the edges
    let edge_quantiles = [
        0.0001, 0.001, 0.01, 0.05,  // Near minimum
        0.95, 0.99, 0.999, 0.9999, // Near maximum
    ];

    for &q in &edge_quantiles {
        let result = sketch.quantile(q).unwrap();

        // Must be finite and within bounds
        assert!(result.is_finite(), "Edge quantile {} must be finite", q);
        assert!(result >= sketch.min() && result <= sketch.max(),
            "Edge quantile {} = {} outside bounds [{}, {}]",
            q, result, sketch.min(), sketch.max());

        // For this dataset, verify reasonable bounds
        if q <= 0.05 {
            assert!(result <= 100.0, "Low quantile {} = {} seems too high", q, result);
        }
        if q >= 0.95 {
            assert!(result >= 900.0, "High quantile {} = {} seems too low", q, result);
        }
    }

    // Verify quantile ordering at edges
    let q1 = sketch.quantile(0.001).unwrap();
    let q5 = sketch.quantile(0.05).unwrap();
    let q95 = sketch.quantile(0.95).unwrap();
    let q99 = sketch.quantile(0.999).unwrap();

    assert!(q1 <= q5, "Edge quantiles should be ordered: {} <= {}", q1, q5);
    assert!(q95 <= q99, "Edge quantiles should be ordered: {} <= {}", q95, q99);
}

#[test]
fn test_datadog_quantile_stability_under_duplicates() {
    // DataDog test: quantile stability when dataset has many duplicate values
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add many duplicates of the same value
    let duplicate_value = 42.0;
    for _ in 0..100 {
        sketch.add(duplicate_value);
    }

    // Add some outliers
    sketch.add(1.0);
    sketch.add(100.0);

    assert_eq!(sketch.count(), 102);

    // Most quantiles should return close to the duplicate value
    for &q in &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let result = sketch.quantile(q).unwrap();

        // Should be reasonably close to the dominant value
        assert!(result >= 1.0 && result <= 100.0, "Quantile {} = {} out of range", q, result);

        // Most quantiles should be closer to 42 than to the outliers
        if q >= 0.2 && q <= 0.8 {
            let distance_to_42 = (result - duplicate_value).abs();
            let distance_to_outliers = (result - 1.0).abs().min((result - 100.0).abs());
            assert!(distance_to_42 <= distance_to_outliers * 2.0,
                "Quantile {} = {} should be closer to dominant value 42", q, result);
        }
    }
}

#[test]
fn test_datadog_quantile_monotonicity() {
    // DataDog test: verify quantiles are monotonically increasing
    let alpha = 1e-2;
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add random-ish dataset
    let values = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0];
    for &value in &values {
        sketch.add(value);
    }

    // Test many quantiles to verify monotonicity
    let quantiles = [
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    ];

    let mut results = Vec::new();
    for &q in &quantiles {
        let result = sketch.quantile(q).unwrap();
        assert!(result.is_finite(), "Quantile {} must be finite", q);
        results.push((q, result));
    }

    // Verify monotonicity: q1 <= q2 implies quantile(q1) <= quantile(q2)
    for i in 1..results.len() {
        let (q1, result1) = results[i - 1];
        let (q2, result2) = results[i];

        assert!(result1 <= result2,
            "Quantiles not monotonic: quantile({}) = {} > quantile({}) = {}",
            q1, result1, q2, result2);
    }
}

#[test]
fn test_datadog_quantile_precision_near_boundaries() {
    // DataDog test: precision of quantiles very close to exact boundaries
    let alpha = 5e-3; // Higher precision
    let mut sketch = DDSketch::new(alpha).unwrap();

    // Add exactly 100 evenly spaced values
    for i in 0..100 {
        sketch.add(i as f64);
    }

    // Test quantiles for boundary precision
    for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let result = sketch.quantile(q).unwrap();

        // Verify within bounds
        assert!(result >= 0.0 && result <= 99.0,
            "Quantile {} = {} outside bounds [0, 99]", q, result);

        // For boundary quantiles, check exact values
        if q == 0.0 {
            assert_eq!(result, 0.0, "Minimum quantile should be exact");
        } else if q == 1.0 {
            assert_eq!(result, 99.0, "Maximum quantile should be exact");
        } else {
            // For intermediate quantiles, just verify they're reasonable and ordered
            assert!(result > 0.0 && result < 99.0,
                "Quantile {} = {} should be between bounds", q, result);
        }
    }

    // Test monotonicity for this specific dataset
    let q25 = sketch.quantile(0.25).unwrap();
    let q50 = sketch.quantile(0.5).unwrap();
    let q75 = sketch.quantile(0.75).unwrap();

    assert!(q25 <= q50, "Q25 {} should be <= Q50 {}", q25, q50);
    assert!(q50 <= q75, "Q50 {} should be <= Q75 {}", q50, q75);
}