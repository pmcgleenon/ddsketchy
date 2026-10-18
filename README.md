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
(a lock-free, atomics-based DDSketch), all configured for 1% relative error and
built with `RUSTFLAGS="-C target-cpu=native"`.

### Insert

ddsketchy ships two index-mapping strategies: `Logarithmic` (the default, bit-exact
with Datadog's reference implementation) and an opt-in `Cubic` mapping that replaces
`f64::ln()` with a cubic-interpolated IEEE-754 bit-manipulation approximation, trading
a small amount of mapping precision (still within the configured relative-error bound)
for insert speed:

```rust
use ddsketchy::{DDSketch, IndexMapping};

let sketch = DDSketch::builder(0.01)
    .mapping(IndexMapping::Cubic)
    .build()?;
```

Mean insert time per value (N = 1,000,000 inserts, lower is better):

| Implementation | ns/insert | Throughput |
|---|---|---|
| ddsketchy (`Cubic`) | 10.5 | 95.1 Melem/s |
| ddsketchy (`Logarithmic`, default) | 16.4 | 61.1 Melem/s |
| quantile-sketch | 8.8 | 113.4 Melem/s |
| sketches-ddsketch | 19.7 | 50.8 Melem/s |

### Merge

Merging one sketch into another, both holding values in `1..1_000_000` at 1%
relative error (about 690 bins). Merge cost is set by the bin count, not by how
many values were inserted. The destination is reused across iterations, so this
is the warm steady state of an aggregator folding many sketches into one:

| Implementation | time/merge |
|---|---|
| ddsketchy | 59 ns |
| sketches-ddsketch | 426 ns |
| quantile-sketch | 922 ns |

### Quantile

Querying a 100,000-value sketch (values uniform in `1..1_000_000`):

| Implementation | q=0.5 | q=0.9 | q=0.99 |
|---|---|---|---|
| ddsketchy | 54.2 ns | 37.9 ns | 34.5 ns |
| sketches-ddsketch | 261.3 ns | 272.0 ns | 275.5 ns |
| quantile-sketch | 705.4 ns | 162.4 ns | 159.5 ns |

## References

* [DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees](https://arxiv.org/pdf/1908.10693.pdf) - The original paper describing the DDSketch algorithm

### Known implementations

| Implementation | Language | Concurrency | License | Notes |
|---|---|---|---|---|
| [ddsketchy](https://github.com/pmcgleenon/ddsketchy) (this crate) | Rust | single-threaded | Apache-2.0 OR MIT | Default `Logarithmic` mapping is bit-exact with `sketches-go`; opt-in `Cubic` mapping and a two-pointer quantile scan trade a little precision/direction-guessing for speed (see [Performance](#performance)) |
| [sketches-go](https://github.com/DataDog/sketches-go) | Go | single-threaded | Apache-2.0 | The original reference implementation |
| [libdd-ddsketch](https://crates.io/crates/libdd-ddsketch) | Rust | single-threaded | Apache-2.0 | Datadog's own official Rust port, published as part of [libdatadog](https://github.com/DataDog/libdatadog/tree/main/libdd-ddsketch) |
| [sketches-java](https://github.com/DataDog/sketches-java) | Java | single-threaded | Apache-2.0 | |
| [sketches-py](https://github.com/DataDog/sketches-py) | Python | single-threaded | custom (see repo) | |
| [sketches-ddsketch](https://github.com/mheffner/rust-sketches-ddsketch) | Rust | single-threaded | Apache-2.0 | Direct Rust port of `sketches-go`; used throughout this crate's benchmarks as the Rust reference |
| [quantile-sketch](https://github.com/tomtomwombat/quantile-sketch) | Rust | lock-free (atomics) | Apache-2.0 | Fixed value range chosen at construction (`insert` panics out-of-range); also benchmarked throughout this crate's README |
| [opentelemetry-rust ddsketch aggregator](https://github.com/cecton/opentelemetry-rust/blob/a0899d9ed595343e9d4eb2dbd994235dedb501ed/opentelemetry/src/sdk/metrics/aggregators/ddsketch.rs#L778-L796) | Rust | single-threaded | Apache-2.0 | Embedded aggregator inside an OpenTelemetry fork, not a standalone crate |
