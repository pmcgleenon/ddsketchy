use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sketches_ddsketch::{Config, DDSketch};

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");
    for &size in &[1_000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::new("sketches-ddsketch", size), &size, |b, &_n| {
            let mut sketch = DDSketch::new(Config::defaults());
            let mut rng = StdRng::seed_from_u64(42);
            b.iter(|| {
                let v = rng.gen::<f64>();
                sketch.add(v);
            });
        });
    }
    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_throughput");
    let mut rng = StdRng::seed_from_u64(42);
    let base = {
        let mut s = DDSketch::new(Config::defaults());
        for _ in 0..100_000 { s.add(rng.gen()); }
        s
    };
    let other = base.clone();

    group.bench_function("sketches-ddsketch", |b| b.iter(|| {
        let mut x = base.clone();
        let _ = x.merge(&other);
    }));
    group.finish();
}

criterion_group!(benches, bench_insert, bench_merge);
criterion_main!(benches);

