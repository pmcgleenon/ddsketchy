// Insert throughput of ddsketchy (default Logarithmic mapping and the opt-in
// Cubic mapping) against `sketches-ddsketch` (Datadog's Go-port reference) and
// `quantile-sketch` (a lock-free, atomics-based DDSketch), all configured for
// 1% relative error over the same value range. The 1,000,000-element rows are
// the ones quoted in the README's Performance section.
//
// Not perfectly like-for-like: `quantile-sketch` must be given the value
// range up front and pre-sizes a fixed bucket array for it, and it tracks no
// sum/min/max, whereas ddsketchy and sketches-ddsketch start empty and grow.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ddsketchy::{DDSketch as Ddsketchy, IndexMapping};
use quantile_sketch::ConcurrentDDSketch as QuantileSketch;
use rand::{rngs::StdRng, Rng, SeedableRng};
use sketches_ddsketch::{Config, DDSketch as SketchesDdsketch};

const RELATIVE_ERROR: f64 = 0.01;
const MIN_VALUE: f64 = 1.0;
const MAX_VALUE: f64 = 1_000_000.0;

fn gen_values(seed: u64, size: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| rng.random_range(MIN_VALUE..MAX_VALUE))
        .collect()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for &size in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let values = gen_values(42, size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("ddsketchy-logarithmic", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut s = Ddsketchy::new(RELATIVE_ERROR).expect("valid alpha");
                    for &v in &values {
                        s.add(v);
                    }
                    std::hint::black_box(s);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("ddsketchy-cubic", size), &size, |b, _| {
            b.iter(|| {
                let mut s = Ddsketchy::builder(RELATIVE_ERROR)
                    .mapping(IndexMapping::Cubic)
                    .build()
                    .expect("valid alpha");
                for &v in &values {
                    s.add(v);
                }
                std::hint::black_box(s);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("sketches-ddsketch", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut s = SketchesDdsketch::new(Config::defaults());
                    for &v in &values {
                        s.add(v);
                    }
                    std::hint::black_box(s);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("quantile-sketch", size), &size, |b, _| {
            b.iter(|| {
                let s = QuantileSketch::with_err_and_range(RELATIVE_ERROR, MIN_VALUE, MAX_VALUE);
                for &v in &values {
                    s.insert(v);
                }
                std::hint::black_box(s);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert);
criterion_main!(benches);
