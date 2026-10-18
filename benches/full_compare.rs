// Insert and quantile-query comparison across ddsketchy, sketches-ddsketch,
// and quantile-sketch, all configured for 1% relative error over the same
// value range. This branch (perf-bidirectional-quantile-scan) only changes
// Store::key_at_rank (the quantile-query path); insert is the unmodified
// Logarithmic mapping.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ddsketchy::DDSketch as Ddsketchy;
use quantile_sketch::ConcurrentDDSketch;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal, Pareto};
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

/// Quantile datasets. Where the target bin sits in the bin array depends on
/// the distribution, and the scan cost depends on that position, so a single
/// distribution is not representative:
/// - uniform by value: rank mass piles up in the top few bins (p50 ~94% of
///   the way through the array);
/// - log-uniform: rank and bin index are proportional;
/// - lognormal (sigma=2): mass in the middle of a wide array;
/// - Pareto (alpha=1): mass in the bottom bins with a long sparse tail.
fn quantile_datasets() -> Vec<(&'static str, Vec<f64>)> {
    let lognormal = LogNormal::new(0.0, 2.0).unwrap();
    let pareto = Pareto::new(1.0, 1.0).unwrap();
    let gen = |seed: u64, f: &dyn Fn(&mut StdRng) -> f64| -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..QUANTILE_SIZE).map(|_| f(&mut rng)).collect()
    };
    vec![
        (
            "uniform",
            gen(42, &|r| r.random_range(MIN_VALUE..MAX_VALUE)),
        ),
        (
            "log-uniform",
            gen(42, &|r| MAX_VALUE.powf(r.random::<f64>())),
        ),
        ("lognormal", gen(42, &|r| lognormal.sample(r))),
        ("pareto", gen(42, &|r| pareto.sample(r))),
    ]
}

const QUANTILE_SIZE: usize = 100_000;

fn bench_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile");

    for (dist, values) in quantile_datasets() {
        let mut ddsketchy = Ddsketchy::new(RELATIVE_ERROR).expect("valid alpha");
        let mut sketches_ddsketch = SketchesDdsketch::new(Config::defaults());
        // quantile-sketch needs its value range up front; size it to the data.
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let quantile_sketch = ConcurrentDDSketch::with_err_and_range(RELATIVE_ERROR, lo, hi);
        for &v in &values {
            ddsketchy.add(v);
            sketches_ddsketch.add(v);
            quantile_sketch.insert(v);
        }

        for &q in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            let id = format!("{dist}/{q}");
            group.bench_with_input(BenchmarkId::new("ddsketchy", &id), &q, |b, &q| {
                b.iter(|| std::hint::black_box(ddsketchy.quantile(q).unwrap()));
            });
            group.bench_with_input(BenchmarkId::new("sketches-ddsketch", &id), &q, |b, &q| {
                b.iter(|| std::hint::black_box(sketches_ddsketch.quantile(q).unwrap().unwrap()));
            });
            group.bench_with_input(BenchmarkId::new("quantile-sketch", &id), &q, |b, &q| {
                b.iter(|| std::hint::black_box(quantile_sketch.quantile(q).unwrap()));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_quantile);
criterion_main!(benches);
