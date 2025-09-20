![DDSketchy Logo](assets/sketchy.jpg)

# dd-sketchy

This is a Rust implementation of the [DDSketch](https://arxiv.org/pdf/1908.10693.pdf) quantile sketch algorithm. 
DDSketch is a fully-mergeable quantile sketch with relative-error guarantees and is extremely fast.

# dd-sketchy Features

* Implements the DDSketch algorithm with configurable relative error guarantees
* Optimized for high-throughput data collection scenarios
* Designed for distributed systems with efficient sketch merging

## Implementation Details

dd-sketchy is built with performance and reliability in mind:

* Efficient logarithmic binning scheme
* Automatic memory management with configurable thresholds
* Configurable error bounds for different use cases

## Usage

```rust
use dd_sketchy::{DDSketch, DDSketchError};

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

## Performance

dd-sketchy is optimized for high-throughput scenarios:

* Efficient sample insertion with minimal overhead
* Fast sketch merging for distributed systems
* Memory-efficient storage with automatic bin management

## References

* [DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees](https://arxiv.org/pdf/1908.10693.pdf) - The original paper describing the DDSketch algorithm

## Other implementations

* https://github.com/cecton/opentelemetry-rust/blob/a0899d9ed595343e9d4eb2dbd994235dedb501ed/opentelemetry/src/sdk/metrics/aggregators/ddsketch.rs#L778-L796
* https://github.com/mheffner/rust-sketches-ddsketch
