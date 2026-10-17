// Compares merge() cost between ddsketchy, sketches-ddsketch
// (https://github.com/mheffner/rust-sketches-ddsketch), and quantile-sketch
// (https://github.com/tomtomwombat/quantile-sketch), all configured for 1%
// relative error over the same value range.
//
// quantile-sketch's `ConcurrentDDSketch` has no `Clone` (it owns
// atomically-published heap blocks with a custom `Drop`), so all three
// sketches use `iter_batched` to time only the `merge` call itself,
// excluding setup (clone, or insert-to-rebuild) cost from the measurement.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use ddsketchy::DDSketch;
use quantile_sketch::ConcurrentDDSketch;
use rand::{rngs::StdRng, Rng, SeedableRng};
use sketches_ddsketch::{Config, DDSketch as SketchesDdsketch};

const RELATIVE_ERROR: f64 = 0.01;
const MIN_VALUE: f64 = 1.0;
const MAX_VALUE: f64 = 1_000_000.0;
const SIZE: usize = 100_000;

fn gen_values(seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..SIZE)
        .map(|_| rng.random_range(MIN_VALUE..MAX_VALUE))
        .collect()
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge");
    let values1 = gen_values(42);
    let values2 = gen_values(43);

    {
        let mut s1 = DDSketch::new(RELATIVE_ERROR).unwrap();
        for &v in &values1 {
            s1.add(v);
        }
        let mut s2 = DDSketch::new(RELATIVE_ERROR).unwrap();
        for &v in &values2 {
            s2.add(v);
        }
        group.bench_function("ddsketchy", |b| {
            b.iter_batched(
                || s1.clone(),
                |mut s| {
                    s.merge(&s2).unwrap();
                    s
                },
                BatchSize::SmallInput,
            );
        });
    }

    {
        let mut s1 = SketchesDdsketch::new(Config::defaults());
        for &v in &values1 {
            s1.add(v);
        }
        let mut s2 = SketchesDdsketch::new(Config::defaults());
        for &v in &values2 {
            s2.add(v);
        }
        group.bench_function("sketches-ddsketch", |b| {
            b.iter_batched(
                || s1.clone(),
                |mut s| {
                    let _ = s.merge(&s2);
                    s
                },
                BatchSize::SmallInput,
            );
        });
    }

    {
        let s2 = ConcurrentDDSketch::with_err_and_range(RELATIVE_ERROR, MIN_VALUE, MAX_VALUE);
        for &v in &values2 {
            s2.insert(v);
        }
        group.bench_function("quantile-sketch", |b| {
            b.iter_batched(
                || {
                    let s1 = ConcurrentDDSketch::with_err_and_range(
                        RELATIVE_ERROR,
                        MIN_VALUE,
                        MAX_VALUE,
                    );
                    for &v in &values1 {
                        s1.insert(v);
                    }
                    s1
                },
                |s1| {
                    s1.merge(&s2).unwrap();
                    s1
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_merge);
criterion_main!(benches);
