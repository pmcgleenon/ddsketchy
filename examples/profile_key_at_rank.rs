// Minimal direct harness for Store::key_at_rank (via the public
// positive_key_at_rank passthrough), bypassing Criterion and
// DDSketch::quantile() entirely, to isolate whether a regression is really
// in key_at_rank itself.
use ddsketchy::DDSketch;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let size = 100_000usize;
    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<f64> = (0..size)
        .map(|_| rng.random_range(1.0..1_000_000.0))
        .collect();
    let mut sketch = DDSketch::new(0.01).unwrap();
    for &v in &values {
        sketch.add(v);
    }

    let rank = 49_999u64; // matches q=0.5 for this dataset
    let iters = 20_000_000u64;

    // Warm up.
    for _ in 0..1_000_000 {
        black_box(sketch.positive_key_at_rank(black_box(rank)));
    }

    let start = Instant::now();
    for _ in 0..iters {
        black_box(sketch.positive_key_at_rank(black_box(rank)));
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_secs_f64() * 1e9 / iters as f64;
    println!("rank={rank}: {ns_per_call:.2} ns/call ({iters} iters, {elapsed:?} total)");
}
