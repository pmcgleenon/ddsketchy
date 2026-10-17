// Reads raw little-endian f64 values from stdin (produced by
// scripts/extract_trip_times.py streaming NYC TLC FHVHV trip-duration data)
// and feeds them into ddsketchy, sketches-ddsketch (an independent
// Datadog-algorithm port), and quantile-sketch (a lock-free DDSketch),
// without ever storing the values, so this scales to real datasets far too
// large to hold in memory as a Vec. At the end, reports count and max
// relative disagreement between ddsketchy and sketches-ddsketch across many
// quantile probes — real-world differential validation alongside the
// synthetic large_scale_differential.rs tests. quantile-sketch's quantiles
// are printed alongside for comparison but not included in the pass/fail
// bound, since it's a different (lock-free, fixed-range) implementation
// rather than another DDSketch port of the same reference.
//
// Usage:
//   python3 scripts/extract_trip_times.py file1.parquet file2.parquet... \
//     | ./target/release/examples/nyc_taxi_quantile_validation

use ddsketchy::DDSketch;
use quantile_sketch::ConcurrentDDSketch;
use sketches_ddsketch::{Config, DDSketch as ReferenceDDSketch};
use std::io::{self, Read};

const ALPHA: f64 = 0.01;
const PROBES: usize = 999;
// Must bracket the full range of trip_time seconds seen across all months
// (observed min=0.990, max=240472.982); quantile_sketch::insert panics on
// out-of-range values, so these keep a safety margin rather than matching
// the observed range exactly.
const MIN_VALUE: f64 = 0.5;
const MAX_VALUE: f64 = 300_000.0;

fn main() {
    let mut sketch = DDSketch::new(ALPHA).expect("valid alpha");
    let mut reference = ReferenceDDSketch::new(Config::defaults());
    let quantile_sketch = ConcurrentDDSketch::with_err_and_range(ALPHA, MIN_VALUE, MAX_VALUE);

    let stdin = io::stdin();
    let mut reader = io::BufReader::with_capacity(1 << 20, stdin.lock());
    let mut buf = [0u8; 8];
    let mut count: u64 = 0;
    let mut skipped_nonfinite: u64 = 0;

    loop {
        match reader.read_exact(&mut buf) {
            Ok(()) => {
                let v = f64::from_le_bytes(buf);
                if !v.is_finite() || !(MIN_VALUE..=MAX_VALUE).contains(&v) {
                    skipped_nonfinite += 1;
                    continue;
                }
                sketch.add(v);
                reference.add(v);
                quantile_sketch.insert(v);
                count += 1;
                if count % 100_000_000 == 0 {
                    eprintln!("...{count} values streamed so far");
                }
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("read error: {e}"),
        }
    }

    eprintln!("done: count={count} skipped_nonfinite_or_out_of_range={skipped_nonfinite}");
    eprintln!(
        "ddsketchy: min={:.3} max={:.3} mean={:.3}",
        sketch.min(),
        sketch.max(),
        sketch.mean()
    );

    let mut max_rel_diff = 0.0_f64;
    let mut worst_q = 0.0;
    for i in 1..=PROBES {
        let q = i as f64 / (PROBES + 1) as f64;
        let ours = sketch.quantile(q).unwrap();
        let theirs = reference.quantile(q).unwrap().unwrap();
        if theirs != 0.0 {
            let rel_diff = (ours - theirs).abs() / theirs.abs();
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
                worst_q = q;
            }
        }
    }
    println!(
        "count={count} max_rel_diff_vs_sketches_ddsketch={max_rel_diff:.6} (at q={worst_q:.3}, bound=2*alpha={:.3})",
        2.0 * ALPHA
    );

    for &q in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999] {
        let ours = sketch.quantile(q).unwrap();
        let theirs = reference.quantile(q).unwrap().unwrap();
        let qs = quantile_sketch.quantile(q).unwrap();
        println!(
            "q={q:.3}: ddsketchy={ours:.3}  sketches-ddsketch={theirs:.3}  quantile-sketch={qs:.3}"
        );
    }

    assert!(
        max_rel_diff <= 2.0 * ALPHA + 1e-9,
        "ddsketchy diverged from sketches-ddsketch by more than 2*alpha on real data: {max_rel_diff}"
    );
    println!("OK: within bound");
}
