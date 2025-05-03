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

criterion_group!(benches, bench_insert, bench_merge);
criterion_main!(benches);
