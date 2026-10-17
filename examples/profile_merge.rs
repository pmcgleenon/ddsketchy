// Dedicated profiling target (not Criterion) for `perf record`/cachegrind
// against DDSketch::merge. Each iteration clones a fresh copy of sketch1 and
// merges sketch2 into it, so every iteration exercises a true "cold" merge
// (not accumulating into an ever-growing sketch) — matching the pattern
// used in benches/ddsketchy_bench.rs's bench_merge.
use ddsketchy::DDSketch;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;

fn main() {
    let size: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let repeats: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let mut rng1 = StdRng::seed_from_u64(42);
    let values1: Vec<f64> = (0..size).map(|_| rng1.random()).collect();
    let mut rng2 = StdRng::seed_from_u64(43);
    let values2: Vec<f64> = (0..size).map(|_| rng2.random()).collect();

    let mut sketch1 = DDSketch::new(0.01).expect("valid alpha");
    for &v in &values1 {
        sketch1.add(v);
    }
    let mut sketch2 = DDSketch::new(0.01).expect("valid alpha");
    for &v in &values2 {
        sketch2.add(v);
    }

    for _ in 0..repeats {
        let mut s = black_box(sketch1.clone());
        s.merge(black_box(&sketch2)).unwrap();
        black_box(&s);
    }

    eprintln!("merged {repeats} times, sketch1 size={size}, sketch2 size={size}");
}
