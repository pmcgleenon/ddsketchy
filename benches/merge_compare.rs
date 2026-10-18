// Compares merge() cost between ddsketchy, sketches-ddsketch
// (https://github.com/mheffner/rust-sketches-ddsketch), and quantile-sketch
// (https://github.com/tomtomwombat/quantile-sketch), all configured for 1%
// relative error over the same value range.
//
// Each iteration merges `s2` into the same `s1`. A DDSketch's bin array is
// bounded by the key range, not by how many values it holds, so after the
// first merge every further merge does identical work on the same bins. That
// is the steady state of an aggregator folding many sketches into one, and it
// keeps the destination hot in cache. Cloning a fresh destination per
// iteration (`iter_batched`) instead mostly measures the cache misses on the
// clone: on an EPYC 7R13 that inflates ddsketchy from 59 ns to 144 ns while
// barely moving the slower implementations.
//
// quantile-sketch's merge takes `&self` (atomics), so its `s1` needs no `mut`.

use criterion::{criterion_group, criterion_main, Criterion};
use ddsketchy::DDSketch;
use quantile_sketch::ConcurrentDDSketch;
use rand::{rngs::StdRng, Rng, SeedableRng};
use sketches_ddsketch::{Config, DDSketch as SketchesDdsketch};
use std::hint::black_box;

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
            b.iter(|| {
                s1.merge(black_box(&s2)).unwrap();
            });
        });
        black_box(&s1);
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
            b.iter(|| {
                let _ = s1.merge(black_box(&s2));
            });
        });
        black_box(&s1);
    }

    {
        let s1 = ConcurrentDDSketch::with_err_and_range(RELATIVE_ERROR, MIN_VALUE, MAX_VALUE);
        for &v in &values1 {
            s1.insert(v);
        }
        let s2 = ConcurrentDDSketch::with_err_and_range(RELATIVE_ERROR, MIN_VALUE, MAX_VALUE);
        for &v in &values2 {
            s2.insert(v);
        }
        group.bench_function("quantile-sketch", |b| {
            b.iter(|| {
                s1.merge(black_box(&s2)).unwrap();
            });
        });
        black_box(&s1);
    }

    group.finish();
}

criterion_group!(benches, bench_merge);
criterion_main!(benches);
