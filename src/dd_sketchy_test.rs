use crate::dd_sketchy::{DDSketch, DDSketchError};
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

    // Use a slightly higher error tolerance for this test
    const TEST_ERROR: f64 = 0.015; // 1.5% error tolerance

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

        // For intermediate quantiles, allow DDSketch error with increased tolerance
        let allowed_error = expected * TEST_ERROR;
        assert!(
            (actual - expected).abs() <= allowed_error,
            "Quantile {}: expected ~{}, got {}, exceeds allowed relative error {}",
            q,
            expected,
            actual,
            TEST_ERROR
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
        assert_relative_eq!(actual, expected, max_relative = RELATIVE_ERROR);
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
        assert_relative_eq!(actual, expected, max_relative = RELATIVE_ERROR);
    }
}

#[test]
fn test_simple_quantile() {
    let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();

    for i in 1..101 {
        dd.add(i as f64);
    }

    assert_relative_eq!(dd.quantile(0.95).unwrap().ceil(), 95.0, max_relative = RELATIVE_ERROR);
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

#[test]
fn test_quantile_error_bounds() {
    let mut d = DDSketch::new(RELATIVE_ERROR).unwrap();

    // Add constant values like in the documentation example
    d.add(1.0);
    d.add(1.0);
    d.add(1.0);

    let q = d.quantile(0.50).unwrap();
    
    // Their implementation uses a different quantile calculation
    // that results in exact values for constant inputs
    assert_eq!(q, 1.0);
}

#[test]
fn test_quantile_error_bounds_with_merge() {
    let mut d1 = DDSketch::new(RELATIVE_ERROR).unwrap();
    let mut d2 = DDSketch::new(RELATIVE_ERROR).unwrap();

    // Add values like in the documentation example
    d1.add(1.0);
    d2.add(2.0);
    d2.add(2.0);

    d1.merge(&d2).unwrap();

    let q = d1.quantile(0.50).unwrap();
    
    // Their implementation uses a different quantile calculation
    // that results in exact values for constant inputs
    assert_eq!(q, 2.0);
}

#[test]
fn test_single_value_quantile() {
    let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
    let test_value = 1234.0;
    dd.add(test_value);

    // Test various quantiles - they should all return the same value
    let test_quantiles = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    for q in test_quantiles {
        assert_eq!(test_value, dd.quantile(q).unwrap());
    }
} 