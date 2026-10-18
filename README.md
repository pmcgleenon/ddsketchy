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

Time per `quantile()` call on a 100,000-value sketch. Where the target bin sits in
the bin array depends on the distribution, so four are measured. quantile-sketch's
fixed range is sized to each dataset.

![quantile() latency by distribution and quantile](https://raw.githubusercontent.com/pmcgleenon/ddsketchy/main/assets/quantile_bench.svg)

sketches-ddsketch's quantile times are sensitive to code layout: rebuilding the
same benchmark source on the same machine moved them by up to 1.8x (uniform p50
measured 188 ns in one build and 340 ns in this one). ddsketchy and
quantile-sketch stayed within a few percent across builds. Treat the
sketches-ddsketch bars as one sample from that range.

<details>
<summary>Exact numbers</summary>

**Uniform by value, `1..1_000_000`**

| Implementation | p01 | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| ddsketchy | 62 ns | 45 ns | 34 ns | 32 ns | 26 ns |
| sketches-ddsketch | 232 ns | 295 ns | 340 ns | 357 ns | 359 ns |
| quantile-sketch | 398 ns | 474 ns | 524 ns | 187 ns | 184 ns |

**Log-uniform, `1..1_000_000`**

| Implementation | p01 | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| ddsketchy | 25 ns | 43 ns | 67 ns | 40 ns | 32 ns |
| sketches-ddsketch | 40 ns | 77 ns | 233 ns | 385 ns | 420 ns |
| quantile-sketch | 174 ns | 216 ns | 403 ns | 219 ns | 178 ns |

**Lognormal, μ=0, σ=2**

| Implementation | p01 | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| ddsketchy | 62 ns | 69 ns | 85 ns | 69 ns | 48 ns |
| sketches-ddsketch | 149 ns | 206 ns | 277 ns | 349 ns | 407 ns |
| quantile-sketch | 348 ns | 415 ns | 498 ns | 395 ns | 331 ns |

**Pareto, scale 1, α=1**

| Implementation | p01 | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| ddsketchy | 11 ns | 28 ns | 30 ns | 41 ns | 56 ns |
| sketches-ddsketch | 21 ns | 27 ns | 42 ns | 91 ns | 154 ns |
| quantile-sketch | 160 ns | 163 ns | 182 ns | 498 ns | 422 ns |

</details>

To regenerate the chart and tables:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --bench quantile_compare
python3 scripts/plot_quantile_bench.py --machine "<hardware and flags>" --tables
```

## References

* [DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees](https://arxiv.org/pdf/1908.10693.pdf) - The original paper describing the DDSketch algorithm

### Known implementations

| Implementation | Language | License | Notes |
|---|---|---|---|
| [sketches-go](https://github.com/DataDog/sketches-go) | Go | Apache-2.0 | The original reference implementation |
| [libdd-ddsketch](https://crates.io/crates/libdd-ddsketch) | Rust | Apache-2.0 | Datadog's own official Rust port, published as part of [libdatadog](https://github.com/DataDog/libdatadog/tree/main/libdd-ddsketch) |
| [sketches-java](https://github.com/DataDog/sketches-java) | Java | Apache-2.0 | |
| [sketches-py](https://github.com/DataDog/sketches-py) | Python | custom (see repo) | |
| [sketches-ddsketch](https://github.com/mheffner/rust-sketches-ddsketch) | Rust | Apache-2.0 | Direct Rust port of `sketches-go` |
| [quantile-sketch](https://github.com/tomtomwombat/quantile-sketch) | Rust | Apache-2.0 | Fixed value range chosen at construction |
| [opentelemetry-rust ddsketch aggregator](https://github.com/cecton/opentelemetry-rust/blob/a0899d9ed595343e9d4eb2dbd994235dedb501ed/opentelemetry/src/sdk/metrics/aggregators/ddsketch.rs#L778-L796) | Rust | Apache-2.0 | Embedded aggregator inside an OpenTelemetry fork, not a standalone crate |
