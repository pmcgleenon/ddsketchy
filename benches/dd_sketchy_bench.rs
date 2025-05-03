use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use dd_sketchy::DDSketch;

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");
    for &size in &[1_000, 10_000, 100_000] {
        // Pre-generate random numbers
        let mut rng = StdRng::seed_from_u64(42);
        let values: Vec<f64> = (0..size).map(|_| rng.gen()).collect();
        
        group.bench_with_input(BenchmarkId::new("ddsketchy", size), &size, |b, &_n| {
            let mut sketch = DDSketch::new(0.01).expect("Failed to create sketch");
            b.iter(|| {
                for &v in &values {
                    sketch.add(v);
                }
            });
        });
    }
    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_throughput");
    let mut rng = StdRng::seed_from_u64(42);
    
    // Pre-generate random numbers
    let values1: Vec<f64> = (0..100_000).map(|_| rng.gen()).collect();
    let values2: Vec<f64> = (0..100_000).map(|_| rng.gen()).collect();
    
    // prepare two different sketches
    let sketch1 = {
        let mut s = DDSketch::new(0.01).expect("Failed to create sketch1");
        for &v in &values1 {
            s.add(v);
        }
        s
    };
    let sketch2 = {
        let mut s = DDSketch::new(0.01).expect("Failed to create sketch2");
        for &v in &values2 {
            s.add(v);
        }
        s
    };

    group.bench_function("ddsketchy", |b| b.iter(|| {
        let mut s = sketch1.clone();
        s.merge(&sketch2).unwrap();
    }));
    group.finish();
}

fn bench_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile_throughput");
    let mut rng = StdRng::seed_from_u64(42);
    
    // Pre-generate random numbers and create a sketch
    let values: Vec<f64> = (0..100_000).map(|_| rng.gen()).collect();
    let sketch = {
        let mut s = DDSketch::new(0.01).expect("Failed to create sketch");
        for &v in &values {
            s.add(v);
        }
        s
    };

    // Benchmark different quantile queries
    for &quantile in &[0.0, 0.25, 0.5, 0.75, 0.99, 1.0] {
        group.bench_with_input(BenchmarkId::new("ddsketchy", quantile), &quantile, |b, &q| {
            let mut result = 0.0;
            b.iter(|| {
                result += sketch.quantile(q).unwrap();
                criterion::black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_insert, bench_merge, bench_quantile);
criterion_main!(benches);
