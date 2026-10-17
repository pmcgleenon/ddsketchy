// Builds exactly ONE sketch implementation (selected by argv[1]) from the
// stdin stream produced by scripts/extract_trip_times.py, so its peak RSS
// can be measured in isolation via `/usr/bin/time -v` without other
// sketches sharing the process. Companion to nyc_taxi_quantile_validation.rs,
// which builds all three Rust-side sketches together (fine for a quantile
// comparison, useless for an isolated memory measurement).
//
// Usage:
//   python3 scripts/extract_trip_times.py file*.parquet \
//     | /usr/bin/time -v ./target/release/examples/nyc_taxi_memory <mode>
// where <mode> is one of:
//   ddsketchy            (max_bins=4096, the crate default)
//   ddsketchy-tight       (max_bins=1024)
//   sketches-ddsketch     (max_num_bins=2048, the crate default)
//   sketches-ddsketch-tight (max_num_bins=1024)
//   quantile-sketch       (fixed range [0.5, 300000])

use ddsketchy::DDSketch;
use quantile_sketch::ConcurrentDDSketch;
use sketches_ddsketch::{Config, DDSketch as SketchesDdsketch};
use std::io::{self, Read};

const ALPHA: f64 = 0.01;

enum Mode {
    Ddsketchy(usize),
    SketchesDdsketch(u32),
    QuantileSketch,
}

fn main() {
    let mode_arg = std::env::args().nth(1).unwrap_or_default();
    let mode = match mode_arg.as_str() {
        "ddsketchy" => Mode::Ddsketchy(4096),
        "ddsketchy-tight" => Mode::Ddsketchy(1024),
        "sketches-ddsketch" => Mode::SketchesDdsketch(2048),
        "sketches-ddsketch-tight" => Mode::SketchesDdsketch(1024),
        "quantile-sketch" => Mode::QuantileSketch,
        other => {
            eprintln!("unknown mode {other:?}; expected one of ddsketchy, ddsketchy-tight, sketches-ddsketch, sketches-ddsketch-tight, quantile-sketch");
            std::process::exit(1);
        }
    };

    let mut ddsketchy_sketch = match &mode {
        Mode::Ddsketchy(max_bins) => {
            Some(DDSketch::with_max_bins(ALPHA, *max_bins).expect("valid alpha"))
        }
        _ => None,
    };
    let mut sketches_ddsketch_sketch = match &mode {
        Mode::SketchesDdsketch(max_bins) => {
            Some(SketchesDdsketch::new(Config::new(ALPHA, *max_bins, 1.0e-9)))
        }
        _ => None,
    };
    let quantile_sketch_sketch = match &mode {
        Mode::QuantileSketch => Some(ConcurrentDDSketch::with_err_and_range(
            ALPHA, 0.5, 300_000.0,
        )),
        _ => None,
    };

    let stdin = io::stdin();
    let mut reader = io::BufReader::with_capacity(1 << 20, stdin.lock());
    let mut buf = [0u8; 8];
    let mut count: u64 = 0;

    loop {
        match reader.read_exact(&mut buf) {
            Ok(()) => {
                let v = f64::from_le_bytes(buf);
                if !v.is_finite() || v <= 0.0 {
                    continue;
                }
                if let Some(s) = ddsketchy_sketch.as_mut() {
                    s.add(v);
                }
                if let Some(s) = sketches_ddsketch_sketch.as_mut() {
                    s.add(v);
                }
                if let Some(s) = quantile_sketch_sketch.as_ref() {
                    if (0.5..=300_000.0).contains(&v) {
                        s.insert(v);
                    }
                }
                count += 1;
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("read error: {e}"),
        }
    }

    eprintln!("mode={mode_arg} count={count}");
    if let Some(s) = &ddsketchy_sketch {
        eprintln!("ddsketchy: p50={:.3}", s.quantile(0.5).unwrap());
    }
    if let Some(s) = &sketches_ddsketch_sketch {
        eprintln!(
            "sketches-ddsketch: p50={:.3}",
            s.quantile(0.5).unwrap().unwrap()
        );
    }
    if let Some(s) = &quantile_sketch_sketch {
        eprintln!("quantile-sketch: p50={:.3}", s.quantile(0.5).unwrap());
    }
}
