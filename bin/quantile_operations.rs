use std::time::Instant;
use rand::{rngs::StdRng, Rng, SeedableRng};
use dd_sketchy::DDSketch;

fn main() {
    // Pre-generate random values
    const ITERATIONS: usize = 1_000_000;  // Number of values to add to sketch
    const QUANTILE_ITERATIONS: usize = 1_000_000;  // Number of quantile calculations to perform
    println!("Generating {} random values...", ITERATIONS);
    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<f64> = (0..ITERATIONS).map(|_| rng.gen()).collect();
    
    // Profile quantile operations with different value ranges
    let ranges = [
        (0.0, 1.0),    // Small values
        (1.0, 10.0),   // Medium values
        (10.0, 100.0), // Large values
        (-1.0, 1.0),   // Mixed positive/negative
    ];
    
    // Quantiles to measure
    let quantiles = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999];
    
    for (min, max) in &ranges {
        let mut sketch = DDSketch::new(0.01).expect("Failed to create sketch");
        let scaled_values: Vec<f64> = values.iter()
            .map(|&v| min + v * (max - min))
            .collect();
        
        // Pre-populate the sketch
        println!("\nPre-populating sketch with {} values in range [{:.1}, {:.1}]", 
                ITERATIONS, min, max);
        for &v in &scaled_values {
            sketch.add(v);
        }
        
        println!("Profiling quantile operations...");
        
        // Warm up
        for _ in 0..1000 {
            for &q in &quantiles {
                sketch.quantile(q).unwrap();
            }
        }
        
        // Main measurement
        let start = Instant::now();
        for _ in 0..QUANTILE_ITERATIONS {
            for &q in &quantiles {
                sketch.quantile(q).unwrap();
            }
        }
        let duration = start.elapsed();
        
        let total_ops = QUANTILE_ITERATIONS * quantiles.len();
        println!(
            "Throughput: {:.2} M ops/sec",
            total_ops as f64 / duration.as_secs_f64() / 1_000_000.0
        );
        println!(
            "Average time per quantile: {:.2} ns",
            duration.as_nanos() as f64 / total_ops as f64
        );
        
        // Print actual quantile values for verification
        println!("\nQuantile values:");
        for &q in &quantiles {
            println!("  {:.3}: {:.3}", q, sketch.quantile(q).unwrap());
        }
    }
} 