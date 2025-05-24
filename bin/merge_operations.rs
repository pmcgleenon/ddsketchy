use std::time::Instant;
use rand::{rngs::StdRng, Rng, SeedableRng};
use dd_sketchy::DDSketch;

fn main() {
    // Pre-generate random values
    const ITERATIONS: usize = 1_000_000;  // Reduced since we only need for initialization
    println!("Generating {} random values...", ITERATIONS);
    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<f64> = (0..ITERATIONS).map(|_| rng.gen()).collect();
    
    // Profile merge operations with different sketch sizes
    println!("\nMerge operations:");
    let sizes = [1000, 10_000, 100_000, 1_000_000];
    for &size in &sizes {
        let mut sketch1 = DDSketch::new(0.01).expect("Failed to create sketch1");
        let mut sketch2 = DDSketch::new(0.01).expect("Failed to create sketch2");
        
        // Fill sketches with size elements
        for &v in values.iter().take(size) {
            sketch1.add(v);
            sketch2.add(v);
        }
        
        let start = Instant::now();
        sketch1.merge(&sketch2).unwrap();
        let duration = start.elapsed();
        
        println!(
            "Merge operation (size {}): {:.2} ns total, {:.2} ns/element",
            size,
            duration.as_nanos() as f64,
            duration.as_nanos() as f64 / size as f64
        );
    }
} 