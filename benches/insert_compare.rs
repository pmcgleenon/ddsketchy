use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ddsketchy::DDSketch as Ours;
use rand::{rngs::StdRng, Rng, SeedableRng};
use sketches_ddsketch::{Config, DDSketch as Ref};

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for &size in &[1_000usize, 10_000, 100_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let values: Vec<f64> = (0..size).map(|_| rng.random::<f64>() * 1000.0).collect();

        group.bench_with_input(BenchmarkId::new("ddsketchy", size), &size, |b, &_n| {
            b.iter(|| {
                let mut s = Ours::new(0.01).expect("valid alpha");
                for &v in &values {
                    s.add(v);
                }
                std::hint::black_box(s);
            });
        });

        group.bench_with_input(BenchmarkId::new("reference", size), &size, |b, &_n| {
            b.iter(|| {
                let mut s = Ref::new(Config::defaults());
                for &v in &values {
                    s.add(v);
                }
                std::hint::black_box(s);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert);
criterion_main!(benches);
