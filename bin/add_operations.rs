use std::time::Instant;
use rand::{rngs::StdRng, Rng, SeedableRng};
use dd_sketchy::DDSketch;

fn main() {
    // Pre-generate random values
    const ITERATIONS: usize = 100_000_000;
    println!("Generating {} random values...", ITERATIONS);
    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<f64> = (0..ITERATIONS).map(|_| rng.gen()).collect();
    
    // Profile add operations with different value ranges
    let ranges = [
        (0.0, 1.0),    // Small values
        (1.0, 10.0),   // Medium values
        (10.0, 100.0), // Large values
        (-1.0, 1.0),   // Mixed positive/negative
    ];
    
    for (min, max) in &ranges {
        let mut sketch = DDSketch::new(0.01).expect("Failed to create sketch");
        let scaled_values: Vec<f64> = values.iter()
            .map(|&v| min + v * (max - min))
            .collect();
        
        println!("\nProfiling add operations for range [{:.1}, {:.1}]", min, max);
        
        // Warm up
        for &v in scaled_values.iter().take(1000) {
            sketch.add(v);
        }
        
        // Main measurement
        let start = Instant::now();
        for &v in &scaled_values {
            sketch.add(v);
        }
        let duration = start.elapsed();
        
        println!(
            "Throughput: {:.2} M ops/sec",
            ITERATIONS as f64 / duration.as_secs_f64() / 1_000_000.0
        );
        
        // Verify the sketch
        println!("Final sketch size: {} elements", sketch.count());
    }
} 