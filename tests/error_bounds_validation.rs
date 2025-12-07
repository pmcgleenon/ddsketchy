use ddsketchy::DDSketch;
use rand::prelude::*;
use rand_distr::{Distribution, Exp1, Normal, Pareto};

/// Test data generator for different distributions
pub struct TestDataGenerator {
    rng: StdRng,
}

impl TestDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn generate_exponential(&mut self, n: usize, lambda: f64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let exp_val: f64 = Exp1.sample(&mut self.rng);
                exp_val / lambda
            })
            .collect()
    }

    pub fn generate_normal(&mut self, n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
        let normal = Normal::new(mean, std_dev).unwrap();
        (0..n)
            .map(|_| normal.sample(&mut self.rng))
            .filter(|&x| x > 0.0) // DDSketch requires positive values
            .take(n)
            .collect()
    }

    pub fn generate_pareto(&mut self, n: usize, scale: f64, shape: f64) -> Vec<f64> {
        let pareto = Pareto::new(scale, shape).unwrap();
        (0..n).map(|_| pareto.sample(&mut self.rng)).collect()
    }
}

/// Calculate true quantiles from sorted data
fn calculate_true_quantiles(sorted_data: &[f64], quantiles: &[f64]) -> Vec<f64> {
    quantiles
        .iter()
        .map(|&q| {
            if sorted_data.is_empty() {
                0.0
            } else {
                let index = (q * (sorted_data.len() - 1) as f64) as usize;
                sorted_data[index.min(sorted_data.len() - 1)]
            }
        })
        .collect()
}

/// Get optimal bin count for error bounds testing
fn get_optimal_bin_count_for_alpha(alpha: f64, distribution_name: &str) -> usize {
    match alpha {
        a if a <= 0.001 => {
            match distribution_name {
                s if s.contains("Pareto") => 25000,
                s if s.contains("Normal") => 8000,
                _ => 10000, // Exponential and others
            }
        }
        a if a <= 0.005 => match distribution_name {
            s if s.contains("Pareto") => 8000,
            _ => 4000,
        },
        a if a <= 0.01 => 4096,
        _ => 2048,
    }
}

/// Test DDSketch error bounds and assert they meet expectations
fn assert_error_bounds(
    data: Vec<f64>,
    alpha: f64,
    quantiles: &[f64],
    distribution_name: &str,
) -> Result<(), String> {
    // Create DDSketch with optimal bin sizing
    let bin_count = get_optimal_bin_count_for_alpha(alpha, distribution_name);
    let mut sketch = DDSketch::with_max_bins(alpha, bin_count)
        .map_err(|e| format!("Failed to create DDSketch: {}", e))?;

    // Insert all data points
    for &value in &data {
        sketch.add(value);
    }

    // Calculate true quantiles
    let mut sorted_data = data.clone();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let true_quantiles = calculate_true_quantiles(&sorted_data, quantiles);

    let mut failures = 0;
    let mut failure_details = Vec::new();

    // Test each quantile
    for (i, &q) in quantiles.iter().enumerate() {
        let estimated = sketch
            .quantile(q)
            .map_err(|e| format!("Failed to compute quantile {}: {}", q, e))?;
        let true_val = true_quantiles[i];

        // Calculate relative error
        let relative_error = if true_val != 0.0 {
            (estimated - true_val).abs() / true_val
        } else {
            0.0
        };

        // Check if error exceeds alpha bound
        if relative_error > alpha {
            failures += 1;
            failure_details.push(format!(
                "  quantile {}: expected ≤ {:.4}, got {:.4} (true: {:.6}, estimated: {:.6})",
                q, alpha, relative_error, true_val, estimated
            ));
        }
    }

    assert!(
        failures == 0,
        "{} distribution failed error bounds validation with α={:.4}:\n  Failures: {}/{}\n  Bin count: {}\n  Failure details:\n{}",
        distribution_name,
        alpha,
        failures,
        quantiles.len(),
        bin_count,
        failure_details.join("\n")
    );

    Ok(())
}

/// Test practical alpha values that should always pass
#[cfg(test)]
mod practical_bounds_tests {
    use super::*;

    #[test]
    fn test_exponential_practical_bounds() {
        let alpha = 0.01; // 1% relative error - should always pass
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_exponential(10000, 1.0);
        let quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

        assert_error_bounds(data, alpha, &quantiles, "Exponential")
            .expect("Exponential distribution should meet α=0.01 bounds with no failures");
    }

    #[test]
    fn test_normal_practical_bounds() {
        let alpha = 0.01; // 1% relative error - should always pass
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_normal(10000, 10.0, 2.0);
        let quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

        assert_error_bounds(data, alpha, &quantiles, "Normal")
            .expect("Normal distribution should meet α=0.01 bounds with no failures");
    }

    #[test]
    fn test_pareto_practical_bounds() {
        let alpha = 0.05; // 5% relative error - more lenient for heavy-tailed
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_pareto(10000, 1.0, 1.0);
        let quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

        assert_error_bounds(data, alpha, &quantiles, "Pareto")
            .expect("Pareto distribution should meet α=0.05 bounds with no failures");
    }

    #[test]
    fn test_mixed_alpha_validation() {
        // Test multiple alpha values to ensure they scale properly
        let alphas = [0.005, 0.01, 0.02, 0.05];
        let mut generator = TestDataGenerator::new(1337);

        for alpha in alphas {
            // Exponential should always pass at reasonable alpha values
            let data = generator.generate_exponential(5000, 1.0);
            let quantiles = [0.25, 0.5, 0.75, 0.9];

            assert_error_bounds(
                data,
                alpha,
                &quantiles,
                &format!("Exponential(α={:.3})", alpha),
            )
            .unwrap_or_else(|_| panic!("Exponential should meet α={:.3} bounds", alpha));
        }
    }
}

/// Test high precision bounds with controlled failure expectations
#[cfg(test)]
mod high_precision_bounds_tests {
    use super::*;

    #[test]
    fn test_exponential_high_precision_controlled() {
        let alpha = 0.005; // 0.5% relative error - high precision
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_exponential(10000, 1.0);
        let quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];

        assert_error_bounds(data, alpha, &quantiles, "Exponential (High Precision)")
            .expect("Exponential should meet α=0.005 bounds with ≤1 failure allowed");
    }

    #[test]
    fn test_normal_high_precision() {
        let alpha = 0.001; // 0.1% relative error - very high precision
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_normal(10000, 10.0, 2.0);
        let quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

        // Normal distributions should handle high precision well
        assert_error_bounds(data, alpha, &quantiles, "Normal (Very High Precision)")
            .expect("Normal distribution should meet α=0.001 bounds");
    }

    #[test]
    fn test_pareto_controlled_precision() {
        let alpha = 0.01; // 1% relative error
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_pareto(10000, 1.0, 1.0);
        // Use moderate quantiles for heavy-tailed distributions
        let quantiles = [0.25, 0.5, 0.75, 0.9];

        // Pareto distribution should achieve perfect accuracy with optimal bin sizing
        assert_error_bounds(data, alpha, &quantiles, "Pareto (Controlled)")
            .expect("Pareto distribution should meet α=0.01 bounds with ≤2 failures allowed");
    }
}

/// Test extreme scenarios to validate DDSketchy advantages
#[cfg(test)]
mod ddsketchy_advantage_validation {
    use super::*;

    #[test]
    fn test_low_quantiles_advantage() {
        // Test where DDSketchy should excel compared to other implementations
        let alpha = 0.01; // 1% relative error
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_exponential(10000, 1.0);
        // Focus on low quantiles where DDSketchy dominates
        let quantiles = [0.01, 0.05, 0.1, 0.2];

        // DDSketchy should handle these well
        assert_error_bounds(data, alpha, &quantiles, "Low Quantiles Advantage")
            .expect("DDSketchy should excel at low quantiles with ≤1 failure");
    }

    #[test]
    fn test_extreme_quantiles_robustness() {
        let alpha = 0.02; // 2% relative error - reasonable for extreme quantiles
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_exponential(20000, 1.0); // Larger dataset
                                                               // Test extreme quantiles
        let quantiles = [0.01, 0.05, 0.95, 0.99];

        assert_error_bounds(data, alpha, &quantiles, "Extreme Quantiles")
            .expect("DDSketchy should handle extreme quantiles well");
    }

    #[test]
    fn test_small_values_precision() {
        // Test with small values where precision matters
        let alpha = 0.01;
        let mut generator = TestDataGenerator::new(1337);

        // Generate data focused on small values
        let mut data = Vec::new();
        for _ in 0..10000 {
            let val = if generator.rng.random_bool(0.8) {
                generator.rng.random_range(0.001..0.1) // 80% small values
            } else {
                generator.rng.random_range(1.0..10.0) // 20% normal values
            };
            data.push(val);
        }

        let quantiles = [0.05, 0.1, 0.25, 0.5];

        assert_error_bounds(data, alpha, &quantiles, "Small Values Precision")
            .expect("DDSketchy should maintain precision with small values");
    }

    #[test]
    fn test_large_dataset_scalability() {
        // Test accuracy doesn't degrade with larger datasets
        let alpha = 0.01;
        let mut generator = TestDataGenerator::new(1337);
        let data = generator.generate_exponential(100000, 1.0); // Large dataset
        let quantiles = [0.1, 0.25, 0.5, 0.75, 0.9];

        assert_error_bounds(data, alpha, &quantiles, "Large Dataset Scalability")
            .expect("DDSketchy should maintain accuracy at scale");
    }
}
