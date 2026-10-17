[![Crates.io][crates-badge]][crates-url]
[![MIT / Apache 2.0 licensed][license-badge]][license-url]
[![Build Status][actions-badge]][actions-url]

[crates-badge]: https://img.shields.io/crates/v/ddsketchy.svg
[crates-url]: https://crates.io/crates/ddsketchy
[license-badge]: https://img.shields.io/crates/l/ddsketchy.svg
[license-url]: https://github.com/pmcgleenon/ddsketchy/blob/main/LICENSE
[actions-badge]: https://github.com/pmcgleenon/ddsketchy/actions/workflows/rust.yml/badge.svg
[actions-url]: https://github.com/pmcgleenon/ddsketchy/actions?query=workflow%3Arust+branch%3Amain

![DDSketchy Logo](https://raw.githubusercontent.com/pmcgleenon/ddsketchy/main/assets/sketchy.jpg)

# ddsketchy

[📖 Docs](https://docs.rs/ddsketchy)

This is a Rust implementation of the [DDSketch](https://arxiv.org/pdf/1908.10693.pdf) quantile sketch algorithm. 
DDSketch is a fully-mergeable quantile sketch with relative-error guarantees.

# ddsketchy Features

* Implements the DDSketch algorithm with configurable relative error guarantees
* Optimized for high-throughput data collection scenarios
* Designed for distributed systems with efficient sketch merging

## Usage

```rust
use ddsketchy::{DDSketch, DDSketchError};

fn main() -> Result<(), DDSketchError> {
    // Create a new sketch with 1% relative error
    let mut sketch = DDSketch::new(0.01)?;

    // Add some values
    sketch.add(1.0);
    sketch.add(2.0);
    sketch.add(3.0);
    sketch.add(4.0);
    sketch.add(5.0);

    // Get the 50th percentile (median)
    let median = sketch.quantile(0.5)?;
    println!("Median: {}", median);

    // Get the 90th percentile
    let p90 = sketch.quantile(0.9)?;
    println!("90th percentile: {}", p90);

    Ok(())
}
```

## Serialization Support

ddsketchy supports optional serialization via [serde](https://serde.rs/). **Serialization is disabled by default** to keep the library dependency-free. To enable it, add the `serde` feature

```rust
use ddsketchy::DDSketch;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sketch = DDSketch::new(0.01)?;
    sketch.add(1.0);
    sketch.add(2.0);

    // Serialize to JSON
    let json = serde_json::to_string(&sketch)?;
    println!("Serialized: {}", json);

    // Deserialize from JSON
    let restored: DDSketch = serde_json::from_str(&json)?;

    // Verify the sketch works correctly
    assert_eq!(sketch.count(), restored.count());
    assert_eq!(sketch.quantile(0.5)?, restored.quantile(0.5)?);

    Ok(())
}
```

The serialization handles all internal state including infinity values for min/max bounds in empty sketches. Empty sketches serialize min/max as `null` values, while sketches with data serialize them as numbers.

## Python Bindings

ddsketchy provides Python bindings via [PyO3](https://pyo3.rs/).

### Building the Python wheel

```bash
pip install maturin
maturin build --features python --release
```

### Development install

```bash
maturin develop --features python
```

### Running Python tests

```bash
python -m pytest tests/test_python_bindings.py
```

### Python usage

```python
from ddsketchy import DDSketch

sketch = DDSketch(alpha=0.01)

sketch.add(1.0)
sketch.add(2.0)
sketch.add(3.0)

print(f"Median: {sketch.quantile(0.5)}")
print(f"P99: {sketch.quantile(0.99)}")
print(f"Count: {sketch.count}, Sum: {sketch.sum}, Mean: {sketch.mean}")
```

## Performance

Benchmarked on an AWS `c6a.xlarge` (AMD EPYC 7R13, 4 vCPU) against
[`sketches-ddsketch`](https://github.com/mheffner/rust-sketches-ddsketch) (Datadog's
Go-port reference) and [`quantile-sketch`](https://github.com/tomtomwombat/quantile-sketch)
(a lock-free, atomics-based DDSketch), all configured for 1% relative error, built from
a 100,000-value dataset:

### Quantile

| Implementation | q=0.5 | q=0.9 | q=0.99 |
|---|---|---|---|
| ddsketchy | 54.2 ns | 37.9 ns | 34.5 ns |
| sketches-ddsketch | 261.3 ns | 272.0 ns | 275.5 ns |
| quantile-sketch | 705.4 ns | 162.4 ns | 159.5 ns |

`Store::key_at_rank` scans from both ends of the bucket array at once (an
interleaved two-pointer scan bounded by `min(forward, backward)` cost),
rather than guessing a single direction from the requested quantile's rank.
An earlier version picked a direction using a rank-based heuristic — the same
one `quantile-sketch` uses, which is why its own p50 above is much slower
than its p90/p99: that heuristic assumes rank-space is proportional to
bucket-index space, which breaks down for data whose mass concentrates
unevenly across the log-spaced buckets. The two-pointer scan has no
direction to get wrong.

### Correctness validation at scale

Beyond the existing unit and property tests, `tests/large_scale_differential.rs`
(run with `cargo test --release -- --ignored --nocapture`) adds:
- Exact-ground-truth checks: tens of millions of values per distribution
  (including the pathological uniform-by-value case that broke the rank-based
  heuristic above), compared against a true sorted-array quantile, not another
  sketch.
- Billion-scale streamed checks against `sketches-ddsketch` (an independently
  implemented Datadog-algorithm port), values fed to both sketches one at a
  time and never stored, so memory stays O(bucket count) regardless of N.

`examples/nyc_taxi_quantile_validation.rs` runs the same kind of check against
a real dataset: NYC TLC FHVHV (Uber/Lyft) trip-duration data, streamed via
`scripts/extract_trip_times.py`. Across 1,002,281,387 real trip durations,
ddsketchy and sketches-ddsketch agreed exactly (0.000000 relative difference)
at every probed quantile:

```
$ python3 scripts/extract_trip_times.py <parquet files...> \
    | ./target/release/examples/nyc_taxi_quantile_validation
count=1002281387 max_rel_diff_vs_sketches_ddsketch=0.000000 (bound=2*alpha=0.020)
q=0.500: ddsketchy=944.049  sketches-ddsketch=944.049
q=0.900: ddsketchy=2186.826  sketches-ddsketch=2186.826
q=0.990: ddsketchy=4065.236  sketches-ddsketch=4065.236
```

## References

* [DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees](https://arxiv.org/pdf/1908.10693.pdf) - The original paper describing the DDSketch algorithm

## Other implementations

* https://github.com/cecton/opentelemetry-rust/blob/a0899d9ed595343e9d4eb2dbd994235dedb501ed/opentelemetry/src/sdk/metrics/aggregators/ddsketch.rs#L778-L796
* https://github.com/mheffner/rust-sketches-ddsketch
* https://github.com/DataDog/sketches-go
* https://github.com/DataDog/sketches-java
