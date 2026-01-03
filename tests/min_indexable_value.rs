use ddsketchy::DDSketch;

/// Test suite for validating min_indexable_value/minimum value handling
#[cfg(test)]
mod min_indexable_value_tests {
    use super::*;

    #[test]
    fn test_min_indexable_value_usage_validation() {
        let sketch = DDSketch::new(0.01).unwrap();
        let min_indexable_value = sketch.min_possible(); // MinIndexableValue from mapping

        // Test that min_indexable_value is computed from mapping (not hardcoded)
        // For alpha=0.01: min_indexable_value = f64::MIN_POSITIVE * gamma ≈ 2.27e-308
        assert!(
            min_indexable_value > 0.0 && min_indexable_value < 1e-300,
            "min_indexable_value should be ~2.27e-308, got {}",
            min_indexable_value
        );

        // Test very small values that should theoretically use min_indexable_value
        let mut sketch_small = DDSketch::new(0.01).unwrap();

        // Add values around and below min_indexable_value
        let test_values = vec![
            min_indexable_value * 0.1,  // Below min_indexable_value
            min_indexable_value * 0.5,  // Half of min_indexable_value
            min_indexable_value,        // Exactly min_indexable_value
            min_indexable_value * 1.1,  // Slightly above min_indexable_value
            min_indexable_value * 10.0, // Well above min_indexable_value
        ];

        for &value in &test_values {
            sketch_small.add(value);
        }

        // DataDog semantics: values below min_indexable_value go to zero bucket
        // Values: [key_eps*0.1, key_eps*0.5, key_eps, key_eps*1.1, key_eps*10.0]
        // Condition: value.abs() < min_indexable_value
        // First 2 (0.1x and 0.5x) are < min_indexable_value → zero_count
        // Last 3 (1.0x, 1.1x, 10.0x) are >= min_indexable_value → positive_store
        assert_eq!(sketch_small.count(), 5);
        assert_eq!(
            sketch_small.get_zero_count(),
            2,
            "Values < min_indexable_value should go to zero_count (first 2)"
        );
        assert_eq!(
            sketch_small.positive_store_count(),
            3,
            "Values >= min_indexable_value should go to positive store (last 3)"
        );
    }

    #[test]
    fn test_subnormal_key_mapping() {
        let sketch = DDSketch::new(0.01).unwrap();

        // Test subnormal numbers that cause very negative keys
        let subnormal1 = f64::MIN_POSITIVE / 2.0; // True subnormal
        let subnormal2 = f64::MIN_POSITIVE / 1000.0; // Very small subnormal
        let tiny_value = 1e-300; // Extremely small value

        // Check what keys these values map to
        let key1 = sketch.key(subnormal1);
        let key2 = sketch.key(subnormal2);
        let key3 = sketch.key(tiny_value);

        println!("Subnormal key mappings:");
        println!("  {} -> key {}", subnormal1, key1);
        println!("  {} -> key {}", subnormal2, key2);
        println!("  {} -> key {}", tiny_value, key3);

        // These should be very negative keys
        assert!(
            key1 < -1000,
            "Subnormal should map to very negative key, got {}",
            key1
        );
        assert!(
            key2 < key1,
            "Smaller subnormal should map to more negative key"
        );

        // Note: 1e-300 may not be smaller than subnormals in all cases due to precision
        // The key insight is that all these values create very negative keys
        assert!(
            key3 < -1000,
            "Tiny value should map to very negative key, got {}",
            key3
        );

        // Test that these can be mapped back to representative values
        let repr1 = sketch.value(key1);
        let repr2 = sketch.value(key2);
        let repr3 = sketch.value(key3);

        println!("Representative value mappings:");
        println!("  key {} -> {}", key1, repr1);
        println!("  key {} -> {}", key2, repr2);
        println!("  key {} -> {}", key3, repr3);

        // Representative values should be positive and finite
        assert!(
            repr1.is_finite() && repr1 > 0.0,
            "Representative value should be positive finite"
        );
        assert!(
            repr2.is_finite() && repr2 > 0.0,
            "Representative value should be positive finite"
        );
        assert!(
            repr3.is_finite() && repr3 > 0.0,
            "Representative value should be positive finite"
        );
    }

    #[test]
    fn test_pathological_small_values() {
        let mut sketch = DDSketch::new(0.01).unwrap();

        // Mix of pathological small values with normal values
        let values = vec![
            // Pathological small values
            f64::MIN_POSITIVE / 2.0,
            f64::MIN_POSITIVE / 1000.0,
            1e-300,
            1e-250,
            1e-200,
            // Normal small values
            1e-9, // min_indexable_value
            1e-6,
            1e-3,
            // Regular values
            0.1,
            1.0,
            10.0,
        ];

        for &value in &values {
            sketch.add(value);
        }

        // Test that quantile computation doesn't break
        let quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

        for &q in &quantiles {
            let result = sketch.quantile(q);
            assert!(result.is_ok(), "Quantile {} should compute successfully", q);

            let value = result.unwrap();
            assert!(
                value.is_finite(),
                "Quantile {} should be finite, got {}",
                q,
                value
            );
            assert!(
                value >= 0.0,
                "Quantile {} should be non-negative, got {}",
                q,
                value
            );
        }

        // Test specific quantiles that might be affected by extreme small values
        let p01 = sketch.quantile(0.01).unwrap();
        let p05 = sketch.quantile(0.05).unwrap();
        let median = sketch.quantile(0.5).unwrap();

        println!("Quantiles with pathological values:");
        println!("  P01: {}", p01);
        println!("  P05: {}", p05);
        println!("  P50: {}", median);

        // With MinIndexableValue clamping, very small values go to zero bucket
        // Values: [subnormals (clamped to 0), 1e-9, 1e-6, 1e-3, 0.1, 1.0, 10.0]
        // About 5 clamped to zero, 6 normal values
        // P01 will now be 0.0 (values below min_indexable_value are clamped)
        assert_eq!(p01, 0.0, "P01 should be 0.0 (clamped values)");
        assert_eq!(p05, 0.0, "P05 should be 0.0 (clamped values)");
        // Median should be in the small value range (1e-9 to 1e-3)
        assert!(
            (1e-10..=1e-2).contains(&median),
            "Median should be in small value range, got {}",
            median
        );
    }

    #[test]
    fn test_zero_vs_small_value_distinction() {
        let mut sketch = DDSketch::new(0.01).unwrap();

        // Add exact zero
        sketch.add(0.0);
        sketch.add(0.0);

        // Add very small values (but not zero)
        let min_indexable_value = sketch.min_possible();
        sketch.add(min_indexable_value * 0.01); // Much smaller than min_indexable_value -> zero bucket
        sketch.add(min_indexable_value * 0.1); // Still smaller than min_indexable_value -> zero bucket
        sketch.add(min_indexable_value); // Exactly min_indexable_value -> zero bucket (< not <=)
        sketch.add(f64::MIN_POSITIVE / 2.0); // Subnormal -> zero bucket

        // Check distribution between zero_count and positive store
        // DataDog semantics: value.abs() < min_indexable_value goes to zero bucket
        // 5 values < min_indexable_value (2 zeros + 3 tiny), 1 value == min_indexable_value
        assert_eq!(
            sketch.get_zero_count(),
            5,
            "Exact zeros and values < min_indexable_value go to zero_count"
        );
        assert_eq!(
            sketch.positive_store_count(),
            1,
            "Values >= min_indexable_value go to positive store"
        );
        assert_eq!(sketch.negative_store_count(), 0, "No negative values added");
        assert_eq!(sketch.count(), 6, "Total count should be 6");

        // Test quantiles handle zero vs tiny value distinction
        // With 6 values (5 zeros + 1 positive), rank calculation:
        // q=0.99 -> rank = 0.99 * 5 = 4.95 \u2248 4, still in zero range
        // q=1.0 -> max value, which is the positive value
        let p0 = sketch.quantile(0.0).unwrap(); // Should be 0.0 (from zero_count)
        let p50 = sketch.quantile(0.5).unwrap(); // Should be 0.0 (still in zero range: 5 zeros, 1 normal)
        let p100 = sketch.quantile(1.0).unwrap(); // Should be the min_indexable_value value

        assert_eq!(p0, 0.0, "P0 should be exactly 0.0");
        assert_eq!(p50, 0.0, "P50 should be 0.0 (most values in zero bucket)");
        assert!(
            p100 > 0.0,
            "P100 (max) should be the positive value ~min_indexable_value"
        );
    }

    #[test]
    fn test_store_range_impact() {
        let mut sketch = DDSketch::new(0.01).unwrap();
        let min_indexable_value = sketch.min_possible();

        // Add values that would create a very large key range
        // Note: values below min_indexable_value (2.27e-308) will be clamped to zero bucket
        let values = vec![
            // Small values above min_indexable_value (to avoid clamping)
            1e-200, 1e-100, // Medium values
            1.0,    // Large values (positive keys)
            1e6, 1e12,
        ];

        for &value in &values {
            sketch.add(value);

            // Verify sketch remains functional after each addition
            assert!(sketch.count() > 0);
            assert!(sketch.quantile(0.5).is_ok());
        }

        // Test that extreme key range doesn't break quantile computation
        // Note: DDSketch returns RECONSTRUCTED values from keys, not exact values
        // So we verify that quantiles are in the right order and magnitude

        let p0 = sketch.quantile(0.0).unwrap();
        let p25 = sketch.quantile(0.25).unwrap();
        let p50 = sketch.quantile(0.5).unwrap();
        let p75 = sketch.quantile(0.75).unwrap();
        let p100 = sketch.quantile(1.0).unwrap();

        println!("Quantiles:");
        println!("  P0:   {:e}", p0);
        println!("  P25:  {:e}", p25);
        println!("  P50:  {:e}", p50);
        println!("  P75:  {:e}", p75);
        println!("  P100: {:e}", p100);

        // Verify monotonicity
        assert!(p0 <= p25, "P0 should be <= P25");
        assert!(p25 <= p50, "P25 should be <= P50");
        assert!(p50 <= p75, "P50 should be <= P75");
        assert!(p75 <= p100, "P75 should be <= P100");

        // Verify magnitude ranges (allowing for reconstruction error)
        // P0 should be very small (original: 1e-200), reconstructed can be in 1e-30 to 1e-100 range
        assert!(
            p0 > 0.0 && p0 < 1e-10,
            "P0 should be a very small positive value, got {:e}",
            p0
        );
        assert!(
            (1e11..=1e13).contains(&p100),
            "P100 should be in large value range, got {:e}",
            p100
        );

        println!(
            "min_indexable_value used for clamping: {:e}",
            min_indexable_value
        );
    }
}
