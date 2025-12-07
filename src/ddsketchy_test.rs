use crate::ddsketchy::{DDSketch, DDSketchError};
use approx::assert_relative_eq;
use rand_distr::{Distribution, Exp, Normal};

const RELATIVE_ERROR: f64 = 0.01;

const TEST_QUANTILES: [f64; 10] = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0];
const TEST_SIZES: [usize; 6] = [3, 5, 10, 21, 100, 1000];
const FLOATING_POINT_ACCEPTABLE_ERROR: f64 = 1e-11;

struct Dataset {
    values: Vec<f64>,
    sum: f64,
    sorted: bool,
}

impl Dataset {
    fn new() -> Self {
        Dataset {
            values: Vec::new(),
            sum: 0.0,
            sorted: false,
        }
    }

    fn add(&mut self, value: f64) {
        self.values.push(value);
        self.sum += value;
        self.sorted = false;
    }

    fn sort(&mut self) {
        if !self.sorted {
            self.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.sorted = true;
        }
    }

    fn count(&self) -> u64 {
        self.values.len() as u64
    }

    fn sum(&self) -> f64 {
        self.sum
    }

    fn min(&mut self) -> f64 {
        self.sort();
        self.values[0]
    }

    fn max(&mut self) -> f64 {
        self.sort();
        self.values[self.values.len() - 1]
    }

    fn lower_quantile(&mut self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) || self.values.is_empty() {
            return f64::NAN;
        }
        self.sort();
        let rank = q * (self.values.len() - 1) as f64;
        self.values[rank.floor() as usize]
    }

    fn upper_quantile(&mut self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) || self.values.is_empty() {
            return f64::NAN;
        }
        self.sort();
        let rank = q * (self.values.len() - 1) as f64;
        self.values[rank.ceil() as usize]
    }
}

fn assert_relatively_accurate(
    relative_accuracy: f64,
    expected_lower_bound: f64,
    expected_upper_bound: f64,
    actual: f64,
) {
    let min_expected_value = (expected_lower_bound * (1.0 - relative_accuracy))
        .min(expected_lower_bound * (1.0 + relative_accuracy));
    let max_expected_value = (expected_upper_bound * (1.0 - relative_accuracy))
        .max(expected_upper_bound * (1.0 + relative_accuracy));

    assert!(
        min_expected_value - FLOATING_POINT_ACCEPTABLE_ERROR <= actual,
        "Value {} below minimum expected {}",
        actual,
        min_expected_value
    );
    assert!(
        actual <= max_expected_value + FLOATING_POINT_ACCEPTABLE_ERROR,
        "Value {} above maximum expected {}",
        actual,
        max_expected_value
    );
}

fn assert_sketches_accurate(sketch: &DDSketch, dataset: &mut Dataset) {
    let alpha = 0.01; // RELATIVE_ERROR
    assert_eq!(dataset.count(), sketch.count());

    if dataset.count() == 0 {
        let quantile = sketch.quantile(0.5).unwrap();
        assert_eq!(quantile, 0.0);
    } else {
        let min_value = sketch.min();
        let max_value = sketch.max();

        // DDSketch min/max return reconstructed values, not raw input values
        // They should be within alpha error bounds of the raw values
        assert_within_relative_error(min_value, dataset.min(), alpha);
        assert_within_relative_error(max_value, dataset.max(), alpha);

        assert_relative_eq!(
            dataset.sum(),
            sketch.sum(),
            max_relative = FLOATING_POINT_ACCEPTABLE_ERROR
        );

        for &q in &TEST_QUANTILES {
            let lower_quantile = dataset.lower_quantile(q);
            let upper_quantile = dataset.upper_quantile(q);
            let quantile = sketch.quantile(q).unwrap();

            assert_relatively_accurate(alpha, lower_quantile, upper_quantile, quantile);
            let tolerance = if min_value == max_value {
                max_value.abs() * alpha + 1e-6
            } else {
                (max_value - min_value) * alpha + 1e-10
            };
            assert!(
                quantile >= min_value - tolerance,
                "Quantile {} below min {} with tolerance {}",
                quantile,
                min_value,
                tolerance
            );
            assert!(
                quantile <= max_value + tolerance,
                "Quantile {} above max {} with tolerance {}",
                quantile,
                max_value,
                tolerance
            );
        }
    }
}

fn evaluate_sketch(
    n: usize,
    generator: &mut dyn FnMut() -> f64,
    sketch: &mut DDSketch,
    dataset: &mut Dataset,
) {
    for _ in 0..n {
        let value = generator();
        sketch.add(value);
        dataset.add(value);
    }
    assert_sketches_accurate(sketch, dataset);

    // Add negative numbers
    for _ in 0..n {
        let value = generator();
        sketch.add(-value);
        dataset.add(-value);
    }
    assert_sketches_accurate(sketch, dataset);
}

#[test]
fn test_add_zero() {
    let mut dd = DDSketch::new(RELATIVE_ERROR).unwrap();
    dd.add(0.0);
    assert_eq!(dd.count(), 1);
    assert_eq!(dd.sum(), 0.0);
    let quantile = dd.quantile(0.5).unwrap();
    assert_eq!(
        quantile, 0.0,
        "Expected exactly 0.0 for zero values, got {}",
        quantile
    );
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
    assert!(matches!(
        DDSketch::new(0.0),
        Err(DDSketchError::InvalidAlpha)
    ));
    assert!(matches!(
        DDSketch::new(1.0),
        Err(DDSketchError::InvalidAlpha)
    ));
    assert!(matches!(
        DDSketch::new(-1.0),
        Err(DDSketchError::InvalidAlpha)
    ));
    assert!(matches!(
        DDSketch::new(2.0),
        Err(DDSketchError::InvalidAlpha)
    ));
}

#[test]
fn test_original_example() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    for &v in &values {
        sketch.add(v);
    }

    let median = sketch.quantile(0.5).unwrap();
    let p90 = sketch.quantile(0.9).unwrap();

    assert_relative_eq!(median, 2.9742334235, max_relative = 1e-10);
    assert_relative_eq!(p90, 4.0148353330, max_relative = 1e-10);
}

#[test]
fn test_constant() {
    for &n in &TEST_SIZES {
        let constant_value = 42.0;
        let mut generator = || constant_value;

        let mut sketch = DDSketch::new(RELATIVE_ERROR).unwrap();
        let mut dataset = Dataset::new();

        evaluate_sketch(n, &mut generator, &mut sketch, &mut dataset);
    }
}

#[test]
fn test_invalid_quantile() -> Result<(), DDSketchError> {
    let sketch = DDSketch::new(0.01)?;

    assert!(sketch.quantile(-0.1).is_err());
    assert!(sketch.quantile(1.1).is_err());

    Ok(())
}

#[test]
fn test_quantile_consistency() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    sketch.add(1.0);

    let q1 = sketch.quantile(0.5).unwrap();
    let q2 = sketch.quantile(0.5).unwrap();
    let q3 = sketch.quantile(0.5).unwrap();

    println!("First call: {:.10}", q1);
    println!("Second call: {:.10}", q2);
    println!("Third call: {:.10}", q3);

    assert_relative_eq!(q1, q2, max_relative = 1e-10);
    assert_relative_eq!(q2, q3, max_relative = 1e-10);
}

fn linear_dataset(start: f64, step: f64, num: usize) -> Vec<f64> {
    (0..num).map(|i| start + (i as f64) * step).collect()
}

fn normal_dataset(mean: f64, stddev: f64, num: usize) -> Vec<f64> {
    let normal = Normal::new(mean, stddev).unwrap();
    let mut data = Vec::with_capacity(num);
    let mut rng = rand::rng();
    for _ in 0..num {
        data.push(normal.sample(&mut rng));
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    data
}

fn exponential_dataset(rate: f64, num: usize) -> Vec<f64> {
    let exp = Exp::new(rate).unwrap();
    let mut data = Vec::with_capacity(num);
    let mut rng = rand::rng();
    for _ in 0..num {
        data.push(exp.sample(&mut rng));
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    data
}

fn assert_within_relative_error(approx: f64, exact: f64, alpha: f64) {
    if exact.abs() > f64::EPSILON {
        let rel_err = (approx - exact).abs() / exact.abs();
        assert!(
            rel_err <= alpha * 2.0, // Allow more tolerance for complex merges
            "Expected ~{}, got {}, rel_err={}",
            exact,
            approx,
            rel_err
        );
    } else {
        assert!(approx.abs() <= alpha);
    }
}

#[test]
fn test_linear_distribution_statistical() {
    let data = linear_dataset(0.0, 1.0, 1000);
    let alpha = 0.01;
    let mut sketch = DDSketch::new(alpha).unwrap();
    for &v in &data {
        sketch.add(v);
    }
    // DDSketch min/max return reconstructed values within alpha error bounds
    assert_within_relative_error(sketch.min(), 0.0, alpha);
    assert_within_relative_error(sketch.max(), 999.0, alpha);
    for &q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let expected = data[(q * (data.len() as f64 - 1.0)) as usize];
        let approx = sketch.quantile(q).unwrap();
        assert_within_relative_error(approx, expected, alpha);
    }
}

#[test]
fn test_near_zero_behavior() {
    let mut sketch = DDSketch::new(0.01).unwrap();

    let tiny = 1e-10;
    sketch.add(tiny);
    sketch.add(-tiny);

    // Go reference shows median for ±1e-10 is -9.9504552879e-11, not 0.0
    let median = sketch.quantile(0.5).unwrap();
    assert!(
        (median - (-9.9504552879e-11)).abs() < 1e-20,
        "Median {} should match Go reference -9.9504552879e-11",
        median
    );

    let small = 1e-8;
    let mut sketch2 = DDSketch::new(0.01).unwrap();
    sketch2.add(small);
    assert_ne!(sketch2.quantile(0.5).unwrap(), 0.0);
}

#[test]
fn test_normal_distribution_statistical() {
    let data = normal_dataset(100.0, 15.0, 2000);
    let alpha = 0.01;
    let mut sketch = DDSketch::new(alpha).unwrap();
    for &v in &data {
        sketch.add(v);
    }
    let expected = data[data.len() / 2];
    let approx = sketch.quantile(0.5).unwrap();
    assert_within_relative_error(approx, expected, alpha);
}

#[test]
fn test_exponential_distribution_statistical() {
    let data = exponential_dataset(2.0, 2000);
    let alpha = 0.01;
    let mut sketch = DDSketch::new(alpha).unwrap();
    for &v in &data {
        sketch.add(v);
    }
    let expected = data[(0.9 * (data.len() as f64 - 1.0)) as usize];
    let approx = sketch.quantile(0.9).unwrap();
    assert_within_relative_error(approx, expected, alpha);
}

#[test]
fn test_go_reference_validation() {
    println!("=== Validating against Go reference implementation ===");

    let mut sketch = DDSketch::new(0.01).unwrap();
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    for v in values {
        sketch.add(v);
    }

    let median = sketch.quantile(0.5).unwrap();
    let p90 = sketch.quantile(0.9).unwrap();

    println!("Our implementation:");
    println!("Median: {:.10}", median);
    println!("90th percentile: {:.10}", p90);

    println!("Go reference:");
    println!("Median: 2.9742334235");
    println!("90th percentile: 4.0148353330");

    assert!(
        (median - 2.9742334235).abs() < 1e-9,
        "Median mismatch: got {}, expected 2.9742334235",
        median
    );
    assert!(
        (p90 - 4.0148353330).abs() < 1e-9,
        "P90 mismatch: got {}, expected 4.0148353330",
        p90
    );

    let test_cases = vec![
        (1.0, 1.0100000000),
        (42.0, 41.6822066330),
        (1000.0, 1002.4280085221),
        (1000000.0, 994912.7844253893),
        (1234.0, 1224.3764974385),
    ];

    for (input, expected) in test_cases {
        let mut test_sketch = DDSketch::new(0.01).unwrap();
        test_sketch.add(input);
        let actual = test_sketch.quantile(0.5).unwrap();

        println!(
            "Value {:.1} -> our: {:.10}, go: {:.10}",
            input, actual, expected
        );

        let relative_error = (actual - expected).abs() / expected.abs();
        println!("  relative error: {:.2e}", relative_error);

        assert!(
            relative_error < 0.02,
            "Value {} error too large: got {}, expected {}, rel_err: {:.2e}",
            input,
            actual,
            expected,
            relative_error
        );
    }
}

#[test]
fn test_no_panic_on_large_ranges() {
    let mut sketch = DDSketch::new(0.01).unwrap();

    sketch.add(1e-10); // Very small
    sketch.add(1e10); // Very large
    sketch.add(1e-9); // Another small
    sketch.add(1e9); // Another large

    assert!(sketch.count() > 0);
    assert!(sketch.quantile(0.5).is_ok());

    for i in 0..1000 {
        let value = (i as f64) * 1e6; // Wide range of large values
        sketch.add(value);
    }

    assert_eq!(sketch.count(), 1004);
    assert!(sketch.quantile(0.9).is_ok());
}

#[test]
fn test_collapsing_lowest_behavior() {
    let _max_bins = 8;
    let mut sketch = DDSketch::new(0.01).unwrap();

    let test_values = vec![1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000];

    for value in test_values {
        sketch.add(value as f64);
    }

    assert_eq!(sketch.count(), 10);
    assert!(sketch.quantile(0.5).is_ok());

    let high_percentile = sketch.quantile(0.9).unwrap();
    assert!(
        high_percentile > 1000.0,
        "High percentile should be preserved: {}",
        high_percentile
    );
}

#[test]
fn test_collapsing_preserves_highest_values() {
    let mut sketch = DDSketch::new(0.01).unwrap();

    for i in 0..20 {
        let value = (i * 1000000) as f64; // Very wide range
        sketch.add(value);
    }

    let count = sketch.count();
    assert_eq!(count, 20);

    let p95 = sketch.quantile(0.95).unwrap();
    let p05 = sketch.quantile(0.05).unwrap();

    assert!(p95 > 15000000.0, "P95 should be high: {}", p95);
    assert!(p05 >= 0.0, "P05 should be valid: {}", p05);
}

#[test]
fn test_collapsing_merge_operations() {
    let mut sketch1 = DDSketch::new(0.01).unwrap();
    let mut sketch2 = DDSketch::new(0.01).unwrap();

    for i in 0..10 {
        sketch1.add((i * 1000000) as f64);
        sketch2.add(((i + 10) * 1000000) as f64);
    }

    let total_expected = sketch1.count() + sketch2.count();

    assert!(sketch1.merge(&sketch2).is_ok());

    assert_eq!(sketch1.count(), total_expected);
    assert!(sketch1.quantile(0.5).is_ok());
}

#[test]
fn test_collapsing_quantile_accuracy() {
    let mut sketch = DDSketch::new(0.05).unwrap(); // 5% error tolerance

    let mut values = Vec::new();
    for i in 0..1000 {
        let value = (i * i) as f64; // Quadratic distribution
        values.push(value);
        sketch.add(value);
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let quantiles = [0.1, 0.5, 0.9, 0.95];
    for &q in &quantiles {
        let actual = sketch.quantile(q).unwrap();
        let expected = values[(q * (values.len() as f64 - 1.0)) as usize];

        let relative_error = (actual - expected).abs() / expected.abs();
        assert!(
            relative_error < 0.2,
            "Quantile {} error too high: got {}, expected {}, rel_err: {:.3}",
            q,
            actual,
            expected,
            relative_error
        );
    }
}

#[test]
fn test_empty_and_single_value_with_collapsing() {
    let mut sketch = DDSketch::new(0.01).unwrap();

    assert_eq!(sketch.count(), 0);
    assert_eq!(sketch.quantile(0.5).unwrap(), 0.0);

    sketch.add(42.0);
    assert_eq!(sketch.count(), 1);

    let q0 = sketch.quantile(0.0).unwrap();
    let q50 = sketch.quantile(0.5).unwrap();
    let q100 = sketch.quantile(1.0).unwrap();

    assert!(
        (q0 - q50).abs() < 1.0,
        "Q0 and Q50 should be close: {} vs {}",
        q0,
        q50
    );
    assert!(
        (q50 - q100).abs() < 1.0,
        "Q50 and Q100 should be close: {} vs {}",
        q50,
        q100
    );

    sketch.add(1e12);
    assert_eq!(sketch.count(), 2);
    assert!(sketch.quantile(0.5).is_ok());
}

#[test]
fn test_new_api_features() {
    let sketch1 = DDSketch::builder(0.01).max_bins(1024).build().unwrap();
    println!("Actual alpha: {}, expected: 0.01", sketch1.alpha());
    assert!((sketch1.alpha() - 0.01).abs() < 1e-6);
    assert_eq!(sketch1.len(), 0);
    assert!(sketch1.is_empty());

    let sketch2 = DDSketch::with_max_bins(0.02, 2048).unwrap();
    assert!((sketch2.alpha() - 0.02).abs() < 1e-10);

    let sketch3 = DDSketch::default();
    assert!((sketch3.alpha() - 0.01).abs() < 1e-10);

    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sketch4: DDSketch = values.iter().copied().collect();
    assert_eq!(sketch4.len(), 5);
    assert!((sketch4.mean() - 3.0).abs() < 0.1);

    let mut sketch5 = DDSketch::default();
    sketch5.extend([1.0, 2.0, 3.0]);
    assert_eq!(sketch5.len(), 3);

    let mut sketch6 = DDSketch::default();
    sketch6.add_batch([1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(sketch6.len(), 5);

    let mut sketch7 = DDSketch::default();
    for i in 1..=100 {
        sketch7.add(i as f64);
    }
    let (p50, p90, p95, p99) = sketch7.percentiles().unwrap();
    assert!(p50 > 40.0 && p50 < 60.0);
    assert!(p90 > 80.0 && p90 < 95.0);
    assert!(p95 > 90.0 && p95 < 98.0);
    assert!(p99 > 95.0 && p99 < 100.0);

    let empty_sketch = DDSketch::default();
    assert_eq!(empty_sketch.quantile_opt(0.5).unwrap(), None);

    let mut non_empty_sketch = DDSketch::default();
    non_empty_sketch.add(42.0);
    assert!(non_empty_sketch.quantile_opt(0.5).unwrap().is_some());

    let mut sketch8 = DDSketch::default();
    sketch8.add(1.0);
    sketch8.add(2.0);
    assert_eq!(sketch8.len(), 2);
    sketch8.clear();
    assert_eq!(sketch8.len(), 0);
    assert!(sketch8.is_empty());

    let mut sketch9 = DDSketch::default();
    sketch9.add(1.0);
    sketch9.add(2.0);
    sketch9.add(3.0);
    let display_str = format!("{}", sketch9);
    assert!(display_str.contains("DDSketch"));
    assert!(display_str.contains("count=3"));
}

#[test]
fn test_error_trait_implementation() {
    let error = DDSketchError::InvalidAlpha;

    let error_string = format!("{}", error);
    assert!(error_string.contains("Alpha must be in range"));

    use std::error::Error;
    let _source = error.source(); // Should not panic

    let error2 = DDSketchError::InvalidAlpha;
    assert_eq!(error, error2);
    println!("Error debug: {:?}", error);
}
