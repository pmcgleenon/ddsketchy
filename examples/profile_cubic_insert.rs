// Dedicated profiling target (not Criterion) for `perf record` against the
// IndexMapping::Cubic insert hot path. Repeats a fixed value set many times
// so perf gets a long, steady-state run dominated by `add()` itself rather
// than setup or benchmark-harness overhead.
use ddsketchy::{DDSketch, IndexMapping};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;

fn main() {
    let unique: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let repeats: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000);

    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<f64> = (0..unique)
        .map(|_| rng.random_range(1.0..1_000_000.0))
        .collect();

    let mut sketch = DDSketch::builder(0.01)
        .mapping(IndexMapping::Cubic)
        .build()
        .expect("valid alpha");

    for _ in 0..repeats {
        for &v in &values {
            sketch.add(black_box(v));
        }
    }

    black_box(&sketch);
    eprintln!(
        "inserted {} values ({} unique x {} repeats), count={}",
        unique * repeats,
        unique,
        repeats,
        sketch.count()
    );
}
