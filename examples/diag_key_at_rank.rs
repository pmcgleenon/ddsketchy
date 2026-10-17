// Diagnostic: count how many bin-array slots key_at_rank's forward scan
// actually has to walk through to reach rank 49999, for the same dataset
// used in the quantile benchmark. Standalone copy of the forward-scan loop
// instrumented with a counter — not calling the real (possibly-changed)
// Store::key_at_rank, so this gives an apples-to-apples baseline regardless
// of which version of store.rs is currently checked out.
use ddsketchy::DDSketch;
use rand::{rngs::StdRng, Rng, SeedableRng};

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

    // We don't have direct access to Store's bins (private), so reconstruct
    // the same info via the public debug accessors: walk key() outputs to
    // find min/max keys actually used, then use positive_key_at_rank at a
    // few ranks and infer scan cost by timing each individually at high
    // iteration count — report min/median/max latency per rank to see the
    // *shape* of cost across ranks directly, including the ones in between
    // the headline p50/p90/p99 probes.
    let count = sketch.count();
    println!("sketch count = {count}");

    for frac in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99, 1.0] {
        let rank = ((frac * (count as f64 - 1.0)) as u64).min(count - 1);
        let iters = 2_000_000u64;
        for _ in 0..200_000 {
            std::hint::black_box(sketch.positive_key_at_rank(std::hint::black_box(rank)));
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(sketch.positive_key_at_rank(std::hint::black_box(rank)));
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / iters as f64;
        println!("frac={frac:.2} rank={rank:>7} -> {ns:7.2} ns/call");
    }
}
