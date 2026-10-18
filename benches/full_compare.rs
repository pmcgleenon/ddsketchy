// Insert and quantile-query comparison across ddsketchy, sketches-ddsketch,
// and quantile-sketch, all configured for 1% relative error over the same
// value range. This branch (perf-bidirectional-quantile-scan) only changes
// Store::key_at_rank (the quantile-query path); insert is the unmodified
// Logarithmic mapping.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ddsketchy::DDSketch as Ddsketchy;
use quantile_sketch::ConcurrentDDSketch;
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

        group.bench_with_input(BenchmarkId::new("ddsketchy", size), &size, |b, _| {
            b.iter(|| {
                let mut s = Ddsketchy::new(RELATIVE_ERROR).expect("valid alpha");
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
                let s =
                    ConcurrentDDSketch::with_err_and_range(RELATIVE_ERROR, MIN_VALUE, MAX_VALUE);
                for &v in &values {
                    s.insert(v);
                }
                std::hint::black_box(s);
            });
        });
    }
    group.finish();
}

fn bench_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile");
    let values = gen_values(42, 100_000);

    let ddsketchy = {
        let mut s = Ddsketchy::new(RELATIVE_ERROR).expect("valid alpha");
        for &v in &values {
            s.add(v);
        }
        s
    };
    let sketches_ddsketch = {
        let mut s = SketchesDdsketch::new(Config::defaults());
        for &v in &values {
            s.add(v);
        }
        s
    };
    let quantile_sketch = {
        let s = ConcurrentDDSketch::with_err_and_range(RELATIVE_ERROR, MIN_VALUE, MAX_VALUE);
        for &v in &values {
            s.insert(v);
        }
        s
    };

    for &q in &[0.01, 0.1, 0.5, 0.9, 0.99] {
        group.bench_with_input(BenchmarkId::new("ddsketchy", q), &q, |b, &q| {
            b.iter(|| std::hint::black_box(ddsketchy.quantile(q).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("sketches-ddsketch", q), &q, |b, &q| {
            b.iter(|| std::hint::black_box(sketches_ddsketch.quantile(q).unwrap().unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("quantile-sketch", q), &q, |b, &q| {
            b.iter(|| std::hint::black_box(quantile_sketch.quantile(q).unwrap()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_quantile);
criterion_main!(benches);
